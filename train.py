import numpy as np
import argparse
import cv2
import torch.cuda
import tqdm
import yaml
import torch.nn as nn
from Nets.SketchNet import QueryNet
import os.path as osp
from Datasets.SketchNetDataloader import load_model_sketch_datasets
class Train():
    def __init__(self,cfg):
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        self.channels = 3
        self.pix_size = 224
        self.load_data_train = load_model_sketch_datasets(cfg,'train')
        self.load_data_test = load_model_sketch_datasets(cfg,'test')
        self.net = QueryNet().to(device = self.device)
        self.loss_f = nn.CrossEntropyLoss()
        self.load()
        self.best_clf = 0
        self.opt = torch.optim.Adam(params = self.net.parameters(),lr = 0.00001,betas=(0.5,0.999))
        print("预训练参数加载完成")
    def load(self):
        nets = self.net
        net_status = nets.state_dict()
        pretained_restnet50_status = torch.load(osp.join('params','resnet50-11ad3fa6.pth'))
        keys = pretained_restnet50_status.keys()
        keys = list(keys)
        i = 0
        j = 0
        k = 0
        l = 0
        for name,_ in net_status.items():
            if name.startswith('resnet46'):
                net_status[name] = pretained_restnet50_status[keys[i]]
                i += 1
            if name.startswith('clf_layer1'):
                # print(net_status[name].shape == pretained_restnet50_status[keys[-116+j]].shape)
                net_status[name] = pretained_restnet50_status[keys[-116 + j]]
                j += 1
            if name.startswith('clf_layer2'):
                # print(net_status[name].shape == pretained_restnet50_status[keys[-62 + k]].shape)
                net_status[name] = pretained_restnet50_status[keys[-62 + k]]
                k += 1
        nets.load_state_dict(net_status)
    def train(self,datasets,net,loss_f,opt,epoch):
        lambda_ = 0.001
        pbar = tqdm.tqdm(datasets)
        net.train()
        for meta in pbar:
            l2_reg = torch.tensor(0.0,device=self.device)
            for param in net.parameters():
                l2_reg += torch.norm(param,p = 2) ** 2
            loss3 = lambda_ * l2_reg
            img = meta['sketch_img']
            label_cat = meta['label_cat']
            img = img.reshape(-1,self.channels,self.pix_size,self.pix_size).to(device = self.device)
            label_cat = label_cat.reshape(-1).to(device = self.device)
            clf_emb1,clf_emb2 = net.forward(img)
            loss = loss_f(clf_emb2,label_cat) + loss3
            opt.zero_grad()#优化器梯度清零
            loss.backward()#反向传播
            opt.step()
            torch.cuda.empty_cache()
            info_dict = {'loss':'%.7f' %(loss.item())}
            pbar.set_postfix(info_dict)
            pbar.set_description("Epoch : %d" % (epoch))
    def train_single(self,epochs):
        for epoch in range(epochs):
            self.train(datasets=self.load_data_train,net=self.net,loss_f=self.loss_f,opt=self.opt,epoch=epoch)
            if epoch >= 10 and (epoch % 2) == 0:
                self.test()
    def test(self):
        self.net.eval()
        total_num = 0
        acc_clf_num = 0
        pbar = tqdm.tqdm(self.load_data_test)
        for meta in pbar:
            with torch.no_grad():
                img = meta['sketch_img']
                label_cat = meta['label_cat']
                img = img.reshape(-1,self.channels,self.pix_size,self.pix_size).to(self.device)
                label_cat = label_cat.reshape(-1).to(self.device)
                _,clf_emb2 = self.net(img)
                #概率
                clf_prob = clf_emb2.softmax(dim = 1)
                max_clf_id = clf_prob.argmax(dim = 1)
                sum1 = ((max_clf_id == label_cat) == True).sum()
                total_num += len(img)
                acc_clf_num += sum1
        print("\n模型视图总数: %s, 查询图像中预测类别正确的数量: %d, 查询图像类别正确率: %.4f" % (total_num, acc_clf_num, acc_clf_num / total_num))
        if (acc_clf_num/total_num) > self.best_clf:
            self.best_clf = (acc_clf_num/total_num)
            self.save()
    def save(self):
        print("saving...")
        torch.save(self.net.state_dict(),osp.join('params','backbone','SketchNets.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",type = str,default='./configs/Sketch.yaml',help="Path to (.yaml) config file"
    )

    configargs = parser.parse_args()
    with open(configargs.configs,'r') as f:
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
    # img = torch.randn(3,224,224)
    train = Train(cfg)
    train.train_single(600)

