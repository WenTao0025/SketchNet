import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    def __init__(self,batch_size,hard_or_full,margin):
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size#128
        self.margin = margin#0.2
    def forward(self,feature,label):
        n,m,d = feature.size()
        #print(label.size())
        #print(feature.size())
        #hp_mask是找出所有样本对中具有相同标签的，相同的为true，不同的false
        #正样本，length：62*128*128 两边128个id看是否有相同的
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)
        dist = self.batch_dist(feature)
        mean_dist = dist.mean(1).mean(1)
        dist = dist.view(-1)
        #这是困难样本对发掘，找出每个样本对应的正样本对中最大距离，找出每个样本的每个负样本对中的最小距离，这就相当于进行困难样本挖掘
        hard_hp_dist = torch.max(torch.masked_select(dist,hp_mask).view(n,m,-1),2)[0]
        hard_hn_dist = torch.min(torch.masked_select(dist,hn_mask).view(n,m,-1),2)[0]
        hard_loss_metric = F.relu(self.margin + hard_hp_dist - hard_hn_dist).view(n,-1)
        hard_loss_metric_mean = torch.mean(hard_loss_metric,1)
        #这里是求取所有正负样本对的loss，没有进行困难样本挖掘

        #这里是求取所有正负样本对的loss，没有进行困难样本挖掘
        full_hp__dist = torch.masked_select(dist,hp_mask).view(n,m,-1,1)
        full_hn_dist = torch.masked_select(dist,hn_mask).view(n,m,1,-1)
        full_loss_metric = F.relu(self.margin + full_hp__dist - full_hn_dist).view(n,-1)
        #计算每个正样本对和负样本对之间的triplet loss
        full_loss_metric_sum = full_loss_metric.sum(1)
        full_loss_num = (full_loss_metric != 0).sum(1).float()#对每个条带中loss不为0的样本进行统计
        #计算每个条带的所有triplet loss平均值
        full_loss_metric_mean = full_loss_metric_sum / full_loss_num#Loss不为0的样本采贡献了损失，所以只对贡献的样本进行平均
        full_loss_metric_mean[full_loss_num == 0] = 0
        return full_loss_metric_mean,hard_loss_metric_mean,mean_dist,full_loss_num
    def batch_dist(self,x):
        x2 = torch.sum(x ** 2,2)
        dist = x2.unsqueeze(2) + x2.unsqueeze(2).transpose(1,2) - 2 * torch.matmul(x,x.transpose(1,2))
        dist = torch.sqrt(F.relu(dist))
        return dist







