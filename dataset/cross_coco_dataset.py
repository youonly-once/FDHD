import json
import os
import traceback

import numpy as np
import torch
import unicom
from timm.data.loader import MultiEpochsDataLoader
from torch.utils.data import Dataset
from PIL import Image
from clip import clip
from .utils import pre_caption


class my_dataset(Dataset):

    uni_feature = None
    filename_id_map = None
    labels = None
    @classmethod
    def init_static_var(cls):
        if cls.uni_feature is None:
            cls.uni_feature = np.load(r"F:\datasets\coco\all_uni_featuresls.npy",allow_pickle=True)
        if cls.filename_id_map is None:
            cls.filename_id_map = np.load(r"F:\datasets\coco\filename_id_map.npy",allow_pickle=True)
        if cls.labels is None:
            cls.labels = np.load(r"F:\datasets\coco\labels.npy",allow_pickle=True)

    def __init__(self, root, transform, split,global_var,unicomtransform):
        #self.__class__.init_static_var()
        self.root = root
        self.transform = transform
        self.split = split
        self.max_words = 77
        # self.uni_feature = global_var['uni_feature']
        self.filename_id_map = global_var['filename_id_map']
        self.labels = global_var['labels']
        #self.uni = uni
        self.unicomtransform = unicomtransform

        self.dataPath = os.path.join(self.root, "new_{}.json".format(self.split))
        with open(self.dataPath, "r", encoding="utf8") as f:
            self.dataList = json.load(f)
        # 初始化一个字典来存储新的数据
        new_dict = {}

        # 遍历原始列表
        for entry in self.dataList:
            key = (entry["image_id"], entry["image_path"])
            caption = entry["caption"]
            if key in new_dict:
                # 将新的 caption 添加到已有的 caption 字符串中，用 ; 分隔
                new_dict[key] += caption
            else:
                # 创建新的键值对
                new_dict[key] = caption

        # 生成新的列表字典
        self.dataList = [{"image_id": key[0], "image_path": key[1], "caption": caption} for key, caption in new_dict.items()]

        self.img_ids = {}
        n = 0
        for ann in self.dataList:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        # if self.split == "experiment":
        #     self.split = "train"
        # try:
        #     self.unicom_fea = np.load(os.path.join(self.root, "{}_uni.npy".format(self.split)), allow_pickle=True)
        #     result_dict_train = {index: value for index, value in self.unicom_fea}
        #
        #     self.unicom_fea_test = np.load(os.path.join(self.root, "{}_uni.npy".format('test')), allow_pickle=True)
        #     result_dict_test = {index: value for index, value in self.unicom_fea_test}
        #
        #     result_dict_train.update(result_dict_test)
        #     self.unicom_fea = result_dict_train
        # except Exception as e:
        #     traceback.print_exc()
        #     self.unicom_fea = None

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, index):
        tmpData = self.dataList[index]

        raw_captions = pre_caption(tmpData["caption"], self.max_words)

        caption = clip.tokenize(raw_captions)[0]
        path = tmpData["image_path"]
        path =path.replace('/home/LAB/huanghl/act-test/test_vt/dataset/coco/coco_img_all/COCO_train2014', 'F:/datasets/mscoco/train2014/COCO_train2014')
        path =path.replace('/home/LAB/huanghl/act-test/test_vt/dataset/coco/coco_img_all/COCO_val2014', 'F:/datasets/mscoco/val2014/COCO_val2014')
        im = Image.open(os.path.join(self.root, path))
        #with torch.no_grad():
        im_uni = self.unicomtransform(im).cpu()#.unsqueeze(0).cuda()
            #image_feature= self.uni.forward_features(im_uni).cpu().squeeze(0)
        im = self.transform(im.convert('RGB'))
        #image_feature= self.uni.forward_features(im_uni).cpu()

        #image_feature = self.uni_feature[tmpData["image_id"]]
        # pre_image, idx....
        return im, caption, im_uni, raw_captions, index, self.labels[self.filename_id_map[tmpData["image_id"]]]
        #return im, caption, image_feature, raw_caption, self.img_ids[tmpData["image_id"]]

def load_dataset(path, batch_size,transform,unicomtransform):
    '''
        load datasets : mirflickr, mscoco, nus-wide
    '''

    uni_feature = np.load(r"F:\datasets\coco\all_uni_featuresls.npy", allow_pickle=True)
    filename_id_map = np.load(r"F:\datasets\coco\filename_id_map.npy", allow_pickle=True).item()
    labels = np.load(r"F:\datasets\coco\labels.npy", allow_pickle=True)
    global_var = {'filename_id_map':filename_id_map,'labels':labels}#'uni_feature':uni_feature,
    dataset = {x: my_dataset(path,transform,x,global_var,unicomtransform) for x in
               ['train', 'test', 'val']}



    shuffle = {'train': True, 'test': False, 'val': False}
    drop_last = {'train': True, 'test': True, 'val': True}
    pin = {'train': True, 'test': False, 'val': False}
    dataloader = {x: MultiEpochsDataLoader(dataset[x], batch_size=batch_size, drop_last=drop_last[x], pin_memory=pin[x], shuffle=shuffle[x],
                                num_workers=6) for x in ['train', 'test', 'val']}
    return dataloader['train'],dataloader['test'],dataloader['val']

class cross_coco_test_dataset(Dataset):
    def __init__(self, root, transform=None, split="test", max_words=64):
        self.root = root
        self.transform = transform
        self.split = split
        self.max_words = max_words
        self.dataPath = os.path.join(self.root, "new_{}.json".format(self.split))

        with open(self.dataPath, "r", encoding="utf8") as f:
            """
            [{
                "image_path": "COCO_val2014_000000184613.jpg",
                "image_id": "184613",
                "caption": "A young man holding an umbrella next to a herd of cattle ."
            }, ...]
            """
            self.dataList = json.load(f)
        """
        {
            "<image_id>":{
                "image_path": "COCO_val2014_000000184613.jpg",
                "caption":[//5 captions]
            }
        }
        """
        tmpData = {}
        for val in self.dataList:
            if val.get("image_id") not in tmpData:
                tmpData[val.get("image_id")] = {
                    "image_path": val.get("image_path"), "caption": [pre_caption(val.get("caption"), self.max_words)]}
            else:
                tmpData[val.get("image_id")]["caption"].append(pre_caption(val.get("caption"), self.max_words))
        # sort image_id keys to keep the order of images
        imgIdKeys = sorted(list(tmpData.keys()))
        self.text = []
        self.image = []
        self.img2txt = {}
        self.txt2img = {}
        txt_id = 0
        for id, key in enumerate(imgIdKeys):
            self.image.append(tmpData[key]["image_path"])
            self.img2txt[id] = []
            for tid, caption in enumerate(tmpData[key]["caption"]):
                self.text.append(caption)
                self.img2txt[id].append(txt_id)
                self.txt2img[txt_id] = id
                txt_id += 1

    def preprocess_text(self, textList):
        preCaptionList = clip.tokenize(textList)
        return preCaptionList

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.image[index])
        path =path.replace('/home/LAB/huanghl/act-test/test_vt/dataset/coco/coco_img_all/COCO_train2014', 'F:/datasets/mscoco/train2014/COCO_train2014')
        path =path.replace('/home/LAB/huanghl/act-test/test_vt/dataset/coco/coco_img_all/COCO_val2014', 'F:/datasets/mscoco/val2014/COCO_val2014')
        im = Image.open(path).convert('RGB')
        im = self.transform(im)

        return im, index
