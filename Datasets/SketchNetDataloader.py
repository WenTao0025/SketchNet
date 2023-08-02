import argparse
import glob
import json
from torchvision import transforms
import torch.utils.data
from matplotlib import pyplot as plt
import numpy
import os.path as osp
import os
import pickle
from PIL import Image
import yaml
from torch.utils.data import DataLoader,Dataset
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
class SketchDataset(Dataset):
    def __init__(self,cfg,type):
        self.shuffle = cfg.query_or_model_data.shuffle
        self.batch_size = 6
        self.category_dict = self.read_json()
        self.trans = transforms.ToTensor()
        self.data_paths = self.get_path_list(cfg,type)#得到全部的文件地址
    def __getitem__(self, item):
        data_path = self.data_paths[item]
        data_label = self.get_category(data_path,self.category_dict)
        data_label = torch.tensor(data_label)
        sketch_img = Image.open(data_path)
        sketch_img = self.trans(sketch_img)
        return{'sketch_img':sketch_img,'label_cat':data_label}
    def __len__(self):
        return len(self.data_paths)

    def read_json(self):
        with open("data_dict.json","r") as json_file:
            category_dict = json.load(json_file)
        return category_dict
    def get_path_list(self,cfg,type):
        #得到训练集或测试集所有的查询图像
        path_list = glob.glob("%s/image/SHREC14LSSTB_SKETCHES/*/%s/*.png"%(cfg.query_or_model_data.data_path,type))
        return path_list
    def get_category(self,path_list,category_dict):
       label_cat = -1#找出对应的标签
       for category in category_dict.keys():
           if category in path_list:
                label_cat = category_dict[category]
                break
       return label_cat
def load_model_sketch_datasets(cfg,type):
    type = type
    num_workers = cfg.query_or_model_data.num_workers
    batch_size = cfg.query_or_model_data.batch_size
    shuffle = cfg.query_or_model_data.shuffle
    datasets = SketchDataset(cfg,type)
    datas = DataLoader(datasets, batch_size = batch_size, shuffle = shuffle,num_workers=num_workers,drop_last=True)
    return  datas

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",type = str,default= 'E:\Sketch_based_3D_Retrieval\configs\Sketch.yaml',help='Path to (.yaml) config file.'

    )

    configargs = parser.parse_args()
    with open(configargs.config,'r') as f:
        config = yaml.safe_load(f)

    def dic2namespace(config):
        namespace = argparse.Namespace()
        for key,value in config.items():
            if isinstance(value,dict):
                new_value = dic2namespace(value)
            else:
                new_value = value
            setattr(namespace,key,new_value)
        return namespace

    cfg = dic2namespace(config)
    da = load_model_sketch_datasets(cfg, 'train')
    i = 0
    for dict in da:
        x = dict['sketch_img']
        category = dict['label_cat']
        print(category)
        print(x.shape)

        img = x[0]
        img = img.swapaxes(0, 1)
        img = img.swapaxes(1, 2)
        print(img.shape)
        plt.imshow(img)
        plt.show()

        img2 = x[2]
        img2 = img2.swapaxes(0, 1)
        img2 = img2.swapaxes(1, 2)
        # plt.imshow(img)
        print(img.shape)
        plt.imshow(img2)
        plt.show()
        break
    # path_list = glob.glob("%s/*/%s/*.png" % (cfg.query_or_model_data.data_path, 'train'))
    # print(path_list)


