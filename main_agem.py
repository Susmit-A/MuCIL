import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import chain
import wandb

from dataloader import *
import random
import os
os.environ["WANDB_SILENT"] = "true"

from torch.utils.data import DataLoader
import numpy as np
import argparse
from progbar import Progbar
from copy import deepcopy
from itertools import chain
import gradutils

from transformers import FlavaMultimodalModel, FlavaImageModel, AutoImageProcessor, AutoConfig, FlavaMultimodalConfig, CLIPVisionModelWithProjection, CLIPVisionModel, ViTModel
from fast_transformers.builders import TransformerEncoderBuilder

def set_seeds(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_crossentropy(pred, target, eps=1e-8):
    loss = (target * torch.log(pred + eps)) + ((1 - target)* torch.log(1 - pred + eps))
    return -loss.sum(dim=-1)


def attr_loss(pred, target):
    pred_norms = torch.norm(pred, dim=-1)
    loss = (((1 - target) * pred_norms) / (1 - target).sum(dim=-1, keepdim=True)) - ((target * pred_norms) / target.sum(-1, keepdim=True))
    return loss.mean(), pred_norms[target == 0].mean(), pred_norms[target == 1].mean()


def binary_attr_loss(pred, target):
        assert pred.shape == target.shape
        pred = pred.squeeze(-1)
        nonzero_targets = target.nonzero()
        zero_targets = (1 - target).nonzero()
        
        num_targets = nonzero_targets.shape[0] + zero_targets.shape[0]
        frac_nonzero = nonzero_targets.shape[0] / num_targets
        frac_zero = zero_targets.shape[0] / num_targets
        
        nonzero_targets = target.nonzero(as_tuple=True)
        zero_targets = (1 - target).nonzero(as_tuple=True)
        loss = (1 - frac_nonzero) * F.binary_cross_entropy(pred[nonzero_targets], target[nonzero_targets]) + \
                (1 - frac_zero) * F.binary_cross_entropy(pred[zero_targets], target[zero_targets])
        return loss.mean()


@torch.no_grad()
def nearest_feature_centroid(args, model, concept_label_dict, preloaded_data=None):
    new_exemplar_dict = {}
    class_list = args.current_classes
    for cls in class_list:
        cls_data = [data for data in preloaded_data if data[1] == cls]
        dataset = dataclass([cls], concept_label_dict=concept_label_dict, seed=0, split='train', transforms=transforms, preprocessor=preprocessor, preloaded_data=cls_data, embeddings_file=class_embeddings_file)
        num_exemplars = min(args.class_buffer_size, len(dataset))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
        feats = torch.cat([model(pixel_values=inputs.to('cuda:0')).last_hidden_state.cpu()[:, 0, :] for inputs, _, _ in dataloader], dim=0)
        normed_feats = feats / torch.norm(feats, dim=-1, keepdim=True)
        mean_feat = normed_feats.mean(dim=0).unsqueeze(0)
        distances = (mean_feat - normed_feats).square().sum(dim=-1)
        sorted_indices = torch.argsort(distances)[:num_exemplars]
        new_exemplars = [dataset[i][:2] for i in sorted_indices]
        new_exemplar_dict[cls] = new_exemplars
    return new_exemplar_dict


@torch.no_grad()
def trim_exemplars_priority(args, exemplars):
    """
    Pick the first args.class_buffer_size exemplars per class from the given list of (sorted) exemplars
    """
    return {k: v[:args.class_buffer_size] for k, v in exemplars.items()}


@torch.no_grad()
def trim_exemplars_random(args, exemplars):
    """
    Pick random set of exemplars for each class
    """
    new_exemplars = {}
    for k, v, in exemplars.items():
        if len(v) <= args.class_buffer_size:
            new_exemplars[k] = v
            continue
        random_indices = np.random.choice(np.arange(len(v)), args.class_buffer_size, replace=False)
        new_exemplars[k] = [v[i] for i in random_indices]
    return new_exemplars


@torch.no_grad()
def evaluate_joint(args, mm_model, image_model, concept_projector, concept_classifier, cls_embs, concept_label_dict):
    class_list = args.current_classes
    val_dataset = dataclass(class_list, concept_label_dict=concept_label_dict, transforms=None, preprocessor=preprocessor, split='test', load_embeddings=True, embeddings_file=class_embeddings_file)
    valloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, drop_last=False)

    loss_fn = nn.CrossEntropyLoss()
    attr_loss_fn = attr_loss
    mm_model.eval()
    image_model.eval()

    pbar = Progbar(len(valloader))
    for step, data in enumerate(valloader):
        log_list = []
        imgs, cls, attrs, attr_embs = data
        
        with torch.no_grad():
            feats = image_model(pixel_values=imgs.to('cuda:0')).last_hidden_state
            combined_feats = torch.cat([projector(feats), attr_embs.cuda()], dim=1)

        if args.linear:
            output_feats = mm_model(combined_feats)
        else:
            output_feats = mm_model(hidden_states=combined_feats).last_hidden_state
        
        concepts = output_feats[:, -val_dataset.get_concept_count():, :].cuda()
        concept_values = concept_classifier(concepts)
        concept_reconstructions = concept_projector(concepts)
        concept_alignment_loss = -F.cosine_similarity(concept_reconstructions, attr_embs.cuda()).mean()

        classes = torch.bmm(concepts, cls_embs.repeat(concepts.shape[0], 1, 1)).mean(dim=1)
        loss = loss_fn(classes, cls.cuda()).mean()

        classification_accuracy = sum(torch.argmax(classes.cpu(), dim=-1) == cls) / attr_embs.shape[0]
        log_list.append(('cls acc', classification_accuracy.detach().numpy()))
        log_list.append(('cls loss', loss.cpu().detach().numpy()))

        concept_loss, min_len, max_len = attr_loss_fn(concepts, attrs.to(torch.float32).cuda())
        concept_loss = binary_attr_loss(concept_values, attrs.cuda())
        loss = loss + args.concept_wts * concept_loss.mean() + concept_alignment_loss * 10.0
        concept_accuracy = sum(torch.round(concept_values.cpu()) == torch.round(attrs)) / attr_embs.shape[0]
        log_list.append(('concept acc', concept_accuracy.detach().numpy()))
        log_list.append(('concept loss', concept_loss.cpu().detach().numpy()))
        log_list.append(('high concepts', max_len.cpu().detach().numpy()))
        log_list.append(('low concepts', min_len.cpu().detach().numpy()))
        log_list.append(('concept alignment', concept_alignment_loss.cpu().detach().numpy()))
        
        pbar.update(step + 1, log_list)
    return (np.mean(pbar._values['cls acc'][0] / max(1, pbar._values['cls acc'][1])), 
            np.mean(pbar._values['concept acc'][0] / max(1, pbar._values['concept acc'][1])))

    
def train_joint(args, mm_model, image_model, concept_projector, concept_classifier, cls_embs, concept_label_dict, concept_emb_dict, exemplars):
    if args.experience > 0:
        buffer = list(chain.from_iterable(exemplars.values()))
        print("Current buffer size:", len(buffer))

    class_list = args.current_classes
    bs = args.batch_size if args.experience == 0 else args.batch_size // 2
    exemplars = list(chain.from_iterable(exemplars.values()))
    train_dataset = dataclass(class_list, concept_label_dict=concept_label_dict, transforms=None, preprocessor=preprocessor, load_embeddings=True, embeddings_file=class_embeddings_file)
    trainloader = DataLoader(train_dataset, batch_size=bs, num_workers=8, pin_memory=True, drop_last=False)
    exemplar_dataset = dataclass(class_list, concept_label_dict=concept_label_dict, seed=0, preprocessor=preprocessor, exemplars=exemplars, load_embeddings=True,  embeddings_file=class_embeddings_file)
    exemplarloader = DataLoader(exemplar_dataset, batch_size=bs, num_workers=4, pin_memory=True, drop_last=True)

    print("##################")
    print("Number of samples:", len(train_dataset))
    print("Number of exemplars:", len(exemplar_dataset))
    print("##################")

    opt = optim.Adam(chain(mm_model.parameters(), concept_projector.parameters(), concept_classifier.parameters(), projector.parameters()), lr=args.lr, weight_decay=1e-5)
    schd = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, 1e-4)

    loss_fn = nn.CrossEntropyLoss()
    attr_loss_fn = attr_loss
    image_model.eval()
    pbar = Progbar(len(trainloader))
    best_model_params = deepcopy(mm_model.state_dict())
    best_acc = 0

    if args.experience == 0:
        epochs = args.first_exp_epochs
    else:
        epochs = args.epochs
    for epoch in range(epochs):
        print("Epoch:", epoch)
        mm_model.train()
        pbar = Progbar(len(trainloader))
        iter_erloader = iter(exemplarloader)
        for step, data in enumerate(trainloader):
            log_list = []
            imgs, cls, attrs, attr_embs = data
            num_images = imgs.shape[0]

            if args.experience > 0:
                opt.zero_grad()
                with torch.no_grad():
                    try:
                        er_imgs, er_cls, er_attrs, er_attr_embs = next(iter_erloader)
                    except StopIteration:
                        iter_erloader = iter(exemplarloader)
                        er_imgs, er_cls, er_attrs, er_attr_embs = next(iter_erloader)
                    
                    feats = image_model(pixel_values=er_imgs.to('cuda:0')).last_hidden_state
                combined_feats = torch.cat([projector(feats), er_attr_embs.cuda()], dim=1)
                output_feats = mm_model(hidden_states=combined_feats).last_hidden_state
                concepts = output_feats[:, -train_dataset.get_concept_count():, :].cuda()
                concept_values = concept_classifier(concepts)
                concept_reconstructions = concept_projector(concepts)
                concept_alignment_loss = -F.cosine_similarity(concept_reconstructions, er_attr_embs.cuda()).mean()

                classes = torch.bmm(concepts, cls_embs.repeat(concepts.shape[0], 1, 1)).mean(dim=1)
                loss = loss_fn(classes, er_cls.cuda()).mean()
                concept_loss = binary_attr_loss(concept_values, er_attrs.cuda())
                loss = loss + args.concept_wts * concept_loss.mean() + concept_alignment_loss * args.lambda2
                loss.backward()
                
                past_gradient = gradutils.get_gradient(mm_model)
            
            with torch.no_grad():
                feats = image_model(pixel_values=imgs.to('cuda:0')).last_hidden_state

            try:
                combined_feats = torch.cat([projector(feats), attr_embs.cuda()], dim=1)
            except Exception as e:
                print()
                print(e)
                print(imgs.shape, er_imgs.shape, feats.shape, attr_embs.shape)
                exit()

            opt.zero_grad()

            if args.linear:
                output_feats = mm_model(combined_feats)
            else:
                output_feats = mm_model(hidden_states=combined_feats).last_hidden_state
            
            concepts = output_feats[:, -train_dataset.get_concept_count():, :].cuda()
            concept_values = concept_classifier(concepts)
            concept_reconstructions = concept_projector(concepts)
            concept_alignment_loss = -F.cosine_similarity(concept_reconstructions, attr_embs.cuda()).mean()

            classes = torch.bmm(concepts, cls_embs.repeat(concepts.shape[0], 1, 1)).mean(dim=1)
            loss = loss_fn(classes, cls.cuda()).mean()

            classification_accuracy = sum(torch.argmax(classes.cpu(), dim=-1) == cls) / attr_embs.shape[0]
            log_list.append(('cls acc', classification_accuracy.detach().numpy()))
            log_list.append(('cls loss', loss.cpu().detach().numpy()))

            _, min_len, max_len = attr_loss_fn(concepts, attrs.to(torch.float32).cuda())
            concept_loss = binary_attr_loss(concept_values, attrs.cuda())
            loss = loss + args.concept_wts * concept_loss.mean() + concept_alignment_loss * args.lambda2

            concept_accuracy = sum(torch.round(concept_values.cpu()) == torch.round(attrs)) / attr_embs.shape[0]
            log_list.append(('concept acc', concept_accuracy.detach().numpy()))
            log_list.append(('concept loss', concept_loss.cpu().detach().numpy()))
            log_list.append(('high concepts', max_len.cpu().detach().numpy()))
            log_list.append(('low concepts', min_len.cpu().detach().numpy()))
            log_list.append(('concept alignment', concept_alignment_loss.cpu().detach().numpy()))

            loss.backward()
            if args.experience > 0:
                cur_gradient = gradutils.get_gradient(mm_model)
                dotp = torch.dot(cur_gradient, past_gradient)
                if dotp < 0:
                    ref_mag = torch.dot(past_gradient, past_gradient)
                    new_grad = cur_gradient - ((dotp / ref_mag) * past_gradient)
                    gradutils.update_gradient(mm_model, new_grad)
            opt.step()
            pbar.update(step + 1, log_list)

        if args.linear:
            save_root = 'saved_models/projections_unpaired_linear'
        else:
            save_root = 'saved_models/projections_unpaired'
        save_path = os.path.join(save_root, f'{dataclass.__name__}_{type(image_model).__name__}_{language_model}_exp{args.num_experiences}_projections_rebuttal/{args.log_name}/{args.experience}')
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            'mm_model': mm_model.state_dict(), 
            'projector': concept_projector.state_dict(),
            'classifier': concept_classifier.state_dict()
        }, os.path.join(save_path, f'epoch{epoch}.pth'))
        acc, _ = evaluate_joint(args, mm_model, image_model, concept_projector, concept_classifier, cls_embs, train_dataset.concept_label_dict)
        if acc > best_acc:
            best_acc = acc
            best_model_params = deepcopy(mm_model.state_dict())
        schd.step()
        print()
        del imgs, cls, attrs, attr_embs, feats, combined_feats, output_feats
        if args.experience > 0:
            del er_imgs, er_cls, er_attrs, _

    new_exemplars = None
    if args.experience < args.num_experiences - 1:
        new_exemplars = nearest_feature_centroid(args, image_model, concept_label_dict, preloaded_data=train_dataset.data)

    mm_model.load_state_dict(best_model_params)
    del exemplars
    return train_dataset.concept_label_dict, concept_emb_dict, new_exemplars


def run(args):
    set_seeds(args.seed)
    # Build a transformer encoder

    if args.linear:
        mm_model = TransformerEncoderBuilder.from_kwargs(
            n_layers=2,
            n_heads=12,
            query_dimensions=768//12,
            value_dimensions=768//12,
            feed_forward_dimensions=768,
            attention_type="linear",
            activation="gelu",
            dropout = 0.0,
            attention_dropout = 0.0
        ).get()

    else:
        mm_model_config = FlavaMultimodalConfig(hidden_size=class_emb_size, num_hidden_layers=2, intermediate_size=class_emb_size*3, num_attention_heads=class_emb_size//64)
        mm_model = FlavaMultimodalModel(mm_model_config)
        mm_model.init_weights()
    mm_model.cuda()
    print(mm_model)
    
    image_model = image_model_class.from_pretrained(hface_model).cuda()
    concept_proj = nn.Sequential(
        nn.Linear(class_emb_size, class_emb_size)
    ).cuda()
    concept_cls = nn.Sequential(
        nn.Linear(class_emb_size, 1),
        nn.Sigmoid(),
        nn.Flatten()
    ).cuda()

    class_accuracies = []
    concept_accuracies = []
    class_order = np.arange(NUM_CLASSES)
    class_lists = np.split(class_order, args.num_experiences)
    concept_label_dict = OrderedDict()
    concept_emb_dict = OrderedDict()
    exemplars = {}
    max_acc_dict = {}
    forgetting_dict = {}
    
    class_emb_keys = []

    for i, class_list in enumerate(class_lists):
        args.experience = i
        class_emb_keys.extend(class_list.tolist())
        class_embeddings = torch.stack(list([class_embeddings_dict[k] for k in class_emb_keys]), dim=-1).unsqueeze(0).cuda()
        args.seen_classes = class_emb_keys
        args.current_classes = class_list

        args.class_buffer_size = args.buffer_size // len(class_emb_keys) if args.buffer_size is not None else args.class_buffer_size
        concept_label_dict, concept_emb_dict, new_exemplars = train_joint(args, mm_model, image_model, concept_proj, concept_cls, class_embeddings, concept_label_dict, concept_emb_dict, exemplars=exemplars)
        exemplars = exemplars | new_exemplars if new_exemplars else exemplars

        if args.buffer_size is not None:
            args.class_buffer_size = args.buffer_size // len(class_emb_keys)
            print("Samples per class for next experience:", args.class_buffer_size)
            exemplars = trim_exemplars_priority(args, exemplars)
            buffer = list(chain.from_iterable(exemplars.values()))
            print("Updated buffer size:", len(buffer))

        print("Evaluating on all tasks")
        eval_list = []
        aa_list = []
        for eval_idx in range(i+1):
            args.current_classes = class_lists[eval_idx]
            cls_acc, con_acc = evaluate_joint(args, mm_model, image_model, concept_proj, concept_cls, class_embeddings, concept_label_dict)
            aa_list.append(cls_acc)
            if eval_idx in max_acc_dict.keys():
                if cls_acc > max_acc_dict[eval_idx]:
                    max_acc_dict[eval_idx] = cls_acc

                forgetting_dict[eval_idx] = max_acc_dict[eval_idx] - cls_acc
            else:
                max_acc_dict[eval_idx] = cls_acc

        class_accuracies.append(np.mean(aa_list))
        concept_accuracies.append(con_acc)
        print("Classification accuracy:", class_accuracies[-1])
        print("Concept accuracy:", con_acc)
        print("#########################")

    print("Average accuracies:", class_accuracies)
    print("FAA:", class_accuracies[-1])
    print(forgetting_dict)
    print(concept_accuracies)
    return class_accuracies, concept_accuracies, forgetting_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--concept_wts', type=float, default=5)
    parser.add_argument('--class_buffer_size', type=int, default=50)
    parser.add_argument('--buffer_size', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--first_exp_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--image_encoder', type=str, default='flava', choices=['ViT', 'FLAVA', 'CLIP'])
    parser.add_argument('--text_encoder', type=str, default='flava', choices=['BERT', 'FLAVA', 'CLIP'])
    parser.add_argument('--linear', action="store_true")
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'cub', 'inet100'])
    parser.add_argument('--num_experiences', type=int, default=5)
    parser.add_argument('--log_name', type=str, default=None)
    parser.add_argument('--lambda2', type=float, default=10.0)
    args = parser.parse_args()

    global NUM_CLASSES, dataclass, language_model, embeddings_file, class_embeddings_file, \
        hface_model, image_model_class, projector, preprocessor, class_embeddings_dict, class_emb_size

    wandb.init(project="ContinualConcepts", config=args, name=args.log_name)

    language_model = args.text_encoder
    if language_model == 'CLIP':
        projector = nn.Linear(768, 512).cuda()
    else:
        projector = nn.Identity()

    if args.dataset == 'cifar100':
        NUM_CLASSES = 100
        dataclass = Cifar100Loader
        embeddings_file = f'class_concept_data/cifar100_filtered_new_{language_model}classes_CLS_conceptembs.pth'
        class_embeddings_file = f"class_concept_data/cifar100_filtered_new_{language_model}text_CLS.pth"
    elif args.dataset == 'cub':
        NUM_CLASSES = 200
        dataclass = CUBLoader
        embeddings_file = f'class_concept_data/cub_{language_model}classes_CLS_conceptembs.pth'
        class_embeddings_file = f"class_concept_data/cub_{language_model}text_CLS.pth"
    elif args.dataset == 'inet100':
        NUM_CLASSES = 100
        dataclass = Imagenet100Loader
        embeddings_file = f'class_concept_data/imagenet_{language_model}classes_CLS_conceptembs.pth'
        class_embeddings_file = f"class_concept_data/imagenet_{language_model}text_CLS.pth"

    if args.image_encoder == 'ViT':
        image_model_class = ViTModel
        hface_model = "google/vit-base-patch16-224-in21k"
    elif args.image_encoder == 'CLIP':
        image_model_class = CLIPVisionModel
        hface_model = "openai/clip-vit-base-patch16"
    elif args.image_encoder == 'FLAVA':
        image_model_class = FlavaImageModel
        hface_model = "facebook/flava-full"
    
    preprocessor = AutoImageProcessor.from_pretrained(hface_model)
    class_embeddings_dict = torch.load(embeddings_file)
    class_emb_size = class_embeddings_dict[0].shape[-1]

    class_accuracies, concept_accuracies, forgetting_dict = run(args)
    class_accuracies = {i:v for i, v in enumerate(class_accuracies)}
    concept_accuracies = {i:v for i, v in enumerate(concept_accuracies)}
    wandb.log(data={
        "Average accuracies": class_accuracies,
        "FAA": class_accuracies[len(class_accuracies) - 1],
        "forgetting dict": forgetting_dict,
        "concept accuracies": concept_accuracies
    })
    wandb.finish()

