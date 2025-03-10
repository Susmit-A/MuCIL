import os
import numpy as np
import _pickle as pkl
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import *
from torch.utils.data import Dataset

import json
from collections import OrderedDict
import random
import glob
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


class CUBLoader(Dataset):
    class_concept_emb_dict = None
    concept_emb_dict = None
    concept_emb_dict_full = None

    @staticmethod
    def reset():
        CUBLoader.class_concept_emb_dict = None
        CUBLoader.concept_emb_dict = None
        CUBLoader.concept_emb_dict_full = None
    
    def __init__(
        self,
            incremental_class_list,
            json_file='class_concept_data/cub.json', 
            concept_label_dict = {},
            split='train',
            seed=None, 
            transforms=None,
            preprocessor=None,
            use_all_concepts=False,
            load_embeddings=False,
            concept_emb_dict = OrderedDict(),
            embeddings_file="class_concept_data/cub_FLAVAtext_CLS.pth",
            class_order=list(range(200)),
            preloaded_data=None, # Initialize Dataset from preloaded data of another instance of this class, eg. when searching for exemplars
            exemplars=None,
            dataset_root='/data/ai22mtech12002/projects/ContinualConcepts/CUB_200_2011'
        ):
        """
        class_list: Iterable containing the class labels for which data needs to be sampled
        label_list: Actual labels to be returned

        # pretrained: python baseline_crossmodal_projections.py --epochs 30 --lr 0.003
        # Linear: python baseline_crossmodal_projections.py --epochs 50 --lr 0.0002
        # Full (reinit pretrained): python baseline_crossmodal_projections.py --epochs 30 --lr 0.0003
        """
        super().__init__()
        self.seed = seed
        np.random.seed(seed)
        self.class_concept_dict = json.load(open(json_file, 'r'), object_pairs_hook=OrderedDict)
        if load_embeddings and CUBLoader.class_concept_emb_dict is None:
            class_concept_emb_dict = torch.load(embeddings_file)
            concept_emb_dict = concept_emb_dict
            concept_emb_dict_full = torch.load(embeddings_file.replace('.pth', '_conceptembs.pth'))
            CUBLoader.class_concept_emb_dict = class_concept_emb_dict
            CUBLoader.concept_emb_dict = concept_emb_dict
            CUBLoader.concept_emb_dict_full = concept_emb_dict_full
        self.load_embeddings = load_embeddings
        self.split = split
        self.class_list = self.class_concept_dict.keys()
        self.concept_list = []
        for v in self.class_concept_dict.values():
            self.concept_list += v
        self.concept_list = list(set(self.concept_list))
        self.class_label_map = {i: k for i, k in zip(np.arange(0, 200), self.class_list)}
        self.concept_label_dict = concept_label_dict
        self.class_order = class_order
        self.class_order_map = {k:v for v, k in enumerate(class_order)}
        self.class_order_reverse_map = {k:v for v, k in self.class_order_map.items()}

        if preloaded_data is not None:
            self.data = preloaded_data
            self.is_exemplar_set = exemplars is not None
        else:
            if exemplars is None:
                if transforms is None and preprocessor is None:
                    self.preproc = None
                    if split == 'train':
                        self.transforms = Compose([
                            RandomHorizontalFlip(p=0.5),
                            Resize(304),
                            RandomCrop(299),
                            ToTensor(),
                            Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
                        ])
                    else:
                        self.transforms = Compose([
                            Resize(304),
                            CenterCrop(299),
                            ToTensor(),
                            Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2]) 
                        ])
                elif transforms is not None:
                    self.transforms = transforms
                    self.preproc = None
                else:
                    self.preproc = preprocessor
                    self.transforms = None
                
                self.dataset_root=dataset_root
                pkl_root='CUB_processed_original'
                full_data = pkl.load(open(os.path.join(pkl_root, split + '.pkl'), 'rb'))

                if split == 'train':
                    full_data += pkl.load(open(os.path.join(pkl_root, 'val.pkl'), 'rb'))

                if self.preproc is not None:
                    data = []
                    for item in full_data:
                        if item['class_label'] in incremental_class_list:
                            img_path = item['img_path']
                            img_path = img_path.replace('/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/CUB_200_2011/', '')
                            img_path = Image.open(os.path.join(self.dataset_root, img_path)).convert("RGB")
                            data.append((self.preproc(img_path, return_tensors='pt')['pixel_values'][0], item['class_label']))
                    self.data = data
                else:
                    data = []
                    for item in full_data:
                        if item['class_label'] in incremental_class_list:
                            img_path = item['img_path']
                            img_path = img_path.replace('/juice/scr/scr102/scr/thaonguyen/CUB_supervision/datasets/CUB_200_2011/', '')
                            img_path = Image.open(os.path.join(self.dataset_root, img_path)).convert("RGB")
                            data.append((self.transforms(img_path), item['class_label']))
                    self.data = data
                self.is_exemplar_set = False
            else:
                self.data = exemplars
                self.is_exemplar_set = True
        np.random.shuffle(self.data)

        if not use_all_concepts:
            class_list = incremental_class_list
        else:
            class_list = list(range(len(self.class_concept_dict.keys())))

        for cls in class_list:
            attrs = self.class_concept_dict[self.class_label_map[cls]]
            for attr in attrs:
                if attr not in self.concept_label_dict.keys():
                    self.concept_label_dict[attr] = len(self.concept_label_dict.keys())
                    if load_embeddings:
                        self.concept_emb_dict[attr] = CUBLoader.concept_emb_dict_full[attr]

    def __len__(self):
        return len(self.data)

    def get_concept_count(self):
        return len(self.concept_label_dict)

    def __getitem__(self, index):
        if self.is_exemplar_set:
            image, label = random.choice(self.data)
        else:
            image, label = self.data[index]

        attrs = self.class_concept_dict[self.class_label_map[label]]
        attr_values = []
        
        for attr in attrs:
            attr_values.append(self.concept_label_dict[attr])

        if self.load_embeddings:
            attr_embs = torch.stack(list(self.concept_emb_dict.values()), dim=0)

        concept_vector = np.zeros(self.get_concept_count())
        concept_vector[attr_values] = 1

        if self.load_embeddings:
            return image, label, concept_vector.astype(np.float32), attr_embs
        return image, label, concept_vector.astype(np.float32)


class Cifar100Loader(Dataset):
    class_concept_emb_dict = None
    concept_emb_dict = None
    concept_emb_dict_full = None
    
    def __init__(
            self,
            incremental_class_list,
            json_file='class_concept_data/cifar100_filtered_new.json', 
            concept_label_dict = {},
            split='train',
            seed=None, 
            transforms=None,
            preprocessor=None,
            use_all_concepts=False,
            exemplars=None,
            load_embeddings=False,
            concept_emb_dict = OrderedDict(),
            embeddings_file="class_concept_data/cifar100_filtered_new_FLAVAtext_CLS.pth",
            class_order=list(range(100)),
            preloaded_data=None # Initialize Dataset from preloaded data, eg. when searching for exemplars
        ):
        """
        class_list: Iterable containing the class labels for which data needs to be sampled
        label_list: Actual labels to be returned
        """
        super().__init__()
        np.random.seed(seed)
        self.class_concept_dict = json.load(open(json_file, 'r'), object_pairs_hook=OrderedDict)
        if load_embeddings and Cifar100Loader.class_concept_emb_dict is None:
            self.class_concept_emb_dict = torch.load(embeddings_file)
            self.concept_emb_dict = concept_emb_dict
            concept_emb_dict_full = torch.load(embeddings_file.replace('.pth', '_conceptembs.pth'))
            Cifar100Loader.class_concept_emb_dict = self.class_concept_emb_dict
            Cifar100Loader.concept_emb_dict = self.concept_emb_dict
            Cifar100Loader.concept_emb_dict_full = concept_emb_dict_full
        self.load_embeddings = load_embeddings
        self.split = split
        self.class_list = self.class_concept_dict.keys()
        self.concept_list = []
        for v in self.class_concept_dict.values():
            self.concept_list += v
        self.concept_list = list(set(self.concept_list))
        self.class_label_map = {i: k for i, k in zip(np.arange(0, 100), self.class_list)}
        self.concept_label_dict = concept_label_dict
        self.class_order = class_order
        self.class_order_map = {k:v for v, k in enumerate(class_order)}
        self.class_order_reverse_map = {k:v for v, k in self.class_order_map.items()}

        if exemplars is None:
            if transforms is None and preprocessor is None:
                self.preproc = None
                if split == 'train':
                    self.transforms = Compose([
                        RandomHorizontalFlip(p=0.5),
                        RandomCrop(32, padding=4),
                        ToTensor(),
                        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
                    ])
                else:
                    self.transforms = Compose([
                        ToTensor(),
                        Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
                    ])
            elif transforms is not None:
                self.transforms = transforms
                self.preproc = None
            else:
                self.preproc = preprocessor
                self.transforms = None

            if preloaded_data is None:
                data = torchvision.datasets.CIFAR100(root='.', train=(split=='train'), download=False, transform=self.transforms)
                if self.preproc is not None:
                    self.data = [(self.preproc(item[0], return_tensors='pt')['pixel_values'].squeeze(), item[1]) for item in data if item[1] in incremental_class_list]
                else:
                    self.data = [item for item in data if item[1] in incremental_class_list]
            else:
                self.data = preloaded_data
            self.is_exemplar_set = False
        else:
            if preloaded_data is None:
                self.data = exemplars
            else:
                self.data = preloaded_data
            self.is_exemplar_set = True

        if split == 'train':
            np.random.shuffle(self.data)

        if not use_all_concepts:
            class_list = incremental_class_list
        else:
            class_list = list(range(len(self.class_concept_dict.keys())))

        for cls in class_list:
            attrs = self.class_concept_dict[self.class_label_map[cls]]
            for attr in attrs:
                if attr not in self.concept_label_dict.keys():
                    self.concept_label_dict[attr] = len(self.concept_label_dict.keys())
                    if load_embeddings:
                        self.concept_emb_dict[attr] = Cifar100Loader.concept_emb_dict_full[attr]

    def __len__(self):
        return len(self.data)

    def get_concept_count(self):
        return len(self.concept_label_dict)

    def __getitem__(self, index):
        if self.is_exemplar_set:
            image, label = random.choice(self.data)
        else:
            image, label = self.data[index]
        attrs = self.class_concept_dict[self.class_label_map[label]]
        attr_values = []
        
        for attr in attrs:
            attr_values.append(self.concept_label_dict[attr])

        if self.load_embeddings:
            attr_embs = torch.stack(list(self.concept_emb_dict.values()), dim=0)

        concept_vector = np.zeros(self.get_concept_count())
        concept_vector[attr_values] = 1

        if self.load_embeddings:
            return image, self.class_order_map[label], concept_vector.astype(np.float32), attr_embs
        return image, self.class_order_map[label], concept_vector.astype(np.float32)

    @staticmethod
    def reset():
        Cifar100Loader.class_concept_emb_dict = None
        Cifar100Loader.concept_emb_dict = None
        Cifar100Loader.concept_emb_dict_full = None


class Imagenet100Loader(Dataset):
    def __init__(self,
            incremental_class_list,
            json_file='class_concept_data/imagenet.json', 
            concept_label_dict = {},
            split='train',
            seed=None, 
            transforms=None,
            preprocessor=None,
            use_all_concepts=False,
            exemplars=None,
            load_embeddings=False,
            concept_emb_dict = OrderedDict(),
            embeddings_file=None,
            class_order=list(range(100)),
            data_root = '/data/ai22mtech12002/projects/data/imagenet',
            preloaded_data=None,
            cache_images=False
        ):
        """
        class_list: Iterable containing the class labels for which data needs to be sampled
        label_list: Actual labels to be returned
        """
        super().__init__()
        np.random.seed(seed)

        from imagenet_classes import IMAGENET2012_CLASSES
        dir_idx = {k:v for v, k in enumerate(IMAGENET2012_CLASSES.keys())}
        class_dirs = [d for d in IMAGENET2012_CLASSES.keys() if dir_idx[d] in incremental_class_list]

        self.class_concept_dict = json.load(open(json_file, 'r'), object_pairs_hook=OrderedDict)
        if load_embeddings:
            self.class_concept_emb_dict = torch.load(embeddings_file)
            self.concept_emb_dict = concept_emb_dict
            concept_emb_dict_full = torch.load(embeddings_file.replace('.pth', '_conceptembs.pth'))
        self.load_embeddings = load_embeddings
        self.split = split
        self.class_list = self.class_concept_dict.keys()
        self.concept_list = []
        for v in self.class_concept_dict.values():
            self.concept_list += v
        self.concept_list = list(set(self.concept_list))
        self.class_label_map = {i: k for i, k in zip(np.arange(0, 100), self.class_list)}
        self.concept_label_dict = concept_label_dict
        self.class_order_map = {k:v for v, k in enumerate(class_order)}
        self.class_order_reverse_map = {k:v for v, k in self.class_order_map.items()}

        self.preproc = preprocessor

        if transforms is None and preprocessor is None:
            self.preproc = None
            if split == 'train':
                self.transforms = Compose([
                    Resize(size=256, interpolation=Image.BILINEAR),
                    RandomHorizontalFlip(p=0.5),
                    RandomCrop(224, padding=4),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = Compose([
                    Resize(size=256, interpolation=Image.BILINEAR),
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        if transforms is not None:
            self.transforms = transforms
        if preprocessor is not None:
            self.preproc = preprocessor
            
        if preloaded_data is not None:
            self.data = preloaded_data
            self.is_exemplar_set = exemplars is not None
        else:
            if exemplars is None:
                data = []
                if split == 'train' and cache_images:
                    for i, d in tqdm(enumerate(class_dirs)):
                        label = dir_idx[d]
                        images = glob.glob(os.path.join(data_root, 'train', d, '*.pkl'))
                        if len(images) == 0:
                            images = glob.glob(os.path.join(data_root, 'train', d, '*.JPEG'))
                            for image in images:
                                image = Image.open(image).convert("RGB")
                                image = self.preproc(image, return_tensors='pt')['pixel_values'].squeeze()
                                data.append((image, label))
                        else:
                            images = pkl.load(open(images[0], 'rb'))
                            if self.preproc is not None:
                                images = [self.preproc(image, return_tensors='pt')['pixel_values'].squeeze() for image in images]
                            else:
                                images = [self.transforms(Image.fromarray(image)) for image in images]
                                
                            data.extend(list(zip(images, [label] * len(images))))
                elif split == 'train':
                    for d in class_dirs:
                        label = dir_idx[d]
                        images = glob.glob(os.path.join(data_root, 'train', d, '*.JPEG'))
                        data.extend(list(zip(images, [label] * len(images))))
                else:
                    for d in class_dirs:
                        label = dir_idx[d]
                        images = glob.glob(os.path.join(data_root, 'val', d, '*.JPEG')) + glob.glob(os.path.join(data_root, 'test_set', d, '*.JPEG'))
                        data.extend(list(zip(images, [label] * len(images))))
                
                self.data = data
                self.is_exemplar_set = False
                np.random.shuffle(self.data)
            else:
                self.data = exemplars
                self.is_exemplar_set = True

        if not use_all_concepts:
            class_list = incremental_class_list
        else:
            class_list = list(range(len(self.class_concept_dict.keys())))[:100]

        for cls in class_list:
            attrs = self.class_concept_dict[self.class_label_map[cls]]
            for attr in attrs:
                if attr not in self.concept_label_dict.keys():
                    self.concept_label_dict[attr] = len(self.concept_label_dict.keys())
                    if load_embeddings:
                        self.concept_emb_dict[attr] = concept_emb_dict_full[attr]
    
    def __len__(self):
        return len(self.data)
    
    def get_concept_count(self):
        return len(self.concept_label_dict)

    def __getitem__(self, index):
        if self.is_exemplar_set:
            image, label = random.choice(self.data)
        else:
            image, label = self.data[index]
        if type(image) == str:
            image = Image.open(image).convert("RGB")
            if self.preproc is not None:
                image = self.preproc(image, return_tensors='pt')['pixel_values'].squeeze()
            else:
                image = self.transforms(image)

        attrs = self.class_concept_dict[self.class_label_map[label]]
        attr_values = []
        
        for attr in attrs:
            attr_values.append(self.concept_label_dict[attr])

        if self.load_embeddings:
            attr_embs = torch.stack(list(self.concept_emb_dict.values()), dim=0)

        concept_vector = np.zeros(self.get_concept_count())
        concept_vector[attr_values] = 1

        if self.load_embeddings:
            return image, self.class_order_map[label], concept_vector.astype(np.float32), attr_embs
        return image, self.class_order_map[label], concept_vector.astype(np.float32)
    
    @staticmethod
    def reset():
        pass
