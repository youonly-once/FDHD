import os

import numpy as np
from timm.data.loader import MultiEpochsDataLoader as dataLoader, MultiEpochsDataLoader
from torch.utils.data.dataset import Dataset
import pickle
from torch.utils.data import DataLoader
import torch
from PIL import Image

import evaluation_cvpr


class CustomDataSet(Dataset):
    def __init__(self,  #raw_images ,
    text_token,
    imageu,
    text_raw ,
    labels,
    image_path,transform):
        #self.raw_images = raw_images
        self.text_token = text_token
        self.imageu = imageu
        self.text = text_raw
        self.labels = labels
        self.image_path = image_path
        self.transform = transform

        self.img2txt = {}
        self.txt2img = {}
        txt_id = 0
        for id, key in enumerate(self.image_path):
            self.img2txt[id] = id
            self.txt2img[id] = id



    def __getitem__(self, index):
        #raw_image = self.raw_images[index]
        text_token = self.text_token[index]

        imgu = self.imageu[index]
        t = self.text[index]

        lab = self.labels[index]
        imgpath = self.image_path[index]
        #imgpath=imgpath.replace('E:','F:')
        im = Image.open(imgpath).convert('RGB')
        im = self.transform(im)
        return im, text_token, imgu, t, index,lab

    def __len__(self):
        count = len(self.text_token)
        return count



def load_dataset(dataset, batch_size,transform):
    '''
        load datasets : mirflickr, mscoco, nus-wide
    '''

    train_loc = 'F:/datasets/' + dataset + '/train_l_uni.pkl'
    query_loc = 'F:/datasets/' + dataset + '/query_l_uni.pkl'
    retrieval_loc = 'F:/datasets/' + dataset + '/retrieval_l_uni.pkl'
#_Np todo
    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)

        train_text_token = torch.tensor(data['text_token'], dtype=torch.int64)
       # train_raw_images = torch.tensor(data['image_raw'] ,dtype=torch.float32)

        train_image_u = torch.tensor(data['image_uni'], dtype=torch.float32)
        train_texts_raw = data['text_raw']

        train_labels = torch.tensor(data['label'], dtype=torch.int64)
        train_image_path = data['image_path']
        train_image_path = [s.replace('E:', 'F:') for s in train_image_path]

    with open(query_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        query_text_token = torch.tensor(data['text_token'], dtype=torch.int64)
        #query_raw_images = torch.tensor(data['image_raw'], dtype=torch.float32)

        query_image_u = torch.tensor(data['image_uni'], dtype=torch.float32)
        query_texts_raw = data['text_raw']

        query_labels = torch.tensor(data['label'], dtype=torch.int64)
        query_image_path = data['image_path']
        query_image_path = [s.replace('E:', 'F:') for s in query_image_path]
    with open(retrieval_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        retrieval_text_token = torch.tensor(data['text_token'], dtype=torch.int64)
        #retrieval_raw_images = torch.tensor(data['image_raw'], dtype=torch.float32)

        retrieval_image_u = torch.tensor(data['image_uni'], dtype=torch.float32)
        retrieval_texts_raw = data['text_raw']

        retrieval_labels = torch.tensor(data['label'], dtype=torch.int64)
        retrieval_image_path = data['image_path']
        retrieval_image_path = [s.replace('E:','F:') for s in retrieval_image_path]
   # raw_images = {'train': train_raw_images[:4992], 'query': query_raw_images, 'retrieval': retrieval_raw_images}
    text_token = {'train': train_text_token, 'query': query_text_token, 'retrieval': retrieval_text_token}
    imageu = {'train': train_image_u, 'query': query_image_u, 'retrieval': retrieval_image_u}
    text_raw = {'train': train_texts_raw, 'query': query_texts_raw, 'retrieval': retrieval_texts_raw}
    labels= {'train': train_labels, 'query': query_labels, 'retrieval': retrieval_labels}
    image_path= {'train': train_image_path, 'query': query_image_path, 'retrieval': retrieval_image_path}
    transforms = {'train': transform, 'query': transform, 'retrieval': transform}
    dataset = {x: CustomDataSet(#raw_images=raw_images[x],
                                text_token=text_token[x],
                                imageu=imageu[x], text_raw=text_raw[x],
                                labels=labels[x],image_path=image_path[x],transform=transforms[x]) for x in
               ['train', 'query', 'retrieval']}

    shuffle = {'train': True, 'query': False, 'retrieval': False}
    drop_last = {'train': True, 'query': True, 'retrieval': True}
    pin = {'train': True, 'query': True, 'retrieval': False}
    dataloader = {x: MultiEpochsDataLoader(dataset[x], batch_size=batch_size, drop_last=drop_last[x], pin_memory=pin[x], shuffle=shuffle[x],
                                num_workers=4) for x in ['train', 'query', 'retrieval']}
    return dataloader['train'],dataloader['query'],dataloader['retrieval']
