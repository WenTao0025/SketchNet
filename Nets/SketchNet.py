import torch
import torch.nn as nn
from Nets import BackboneResnet

"""
resnet版后续网络更新，前面公共网络保留了前4个layer和第5个layer的前两个bottleNeck，第5个layer的最后一个bottleNeck及后续
avgpool分别放在方向和类别的分支上，用来提取特征
"""
class QueryNet(nn.Module):
    def __init__(self):
        super(QueryNet, self).__init__()
        self.resnet46 = BackboneResnet.resnet46()

        self.clf_layer1 = torch.nn.Sequential(
            BackboneResnet.Bottleneck(inplanes=1024, planes=256),
            BackboneResnet.Bottleneck(inplanes=1024, planes=256),
            BackboneResnet.Bottleneck(inplanes=1024, planes=256),
        )

        self.clf_layer2 = torch.nn.Sequential(
            BackboneResnet.Bottleneck(inplanes=1024,planes=512,stride=2, downsample=torch.nn.Sequential(
            BackboneResnet.conv1x1(1024, 512 * BackboneResnet.Bottleneck.expansion, 2),
            torch.nn.BatchNorm2d(512 * BackboneResnet.Bottleneck.expansion),
        )),
            BackboneResnet.Bottleneck(inplanes=2048,planes=512),
            BackboneResnet.Bottleneck(inplanes=2048, planes=512),
            torch.nn.AvgPool2d((7, 7), stride=1),
            torch.nn.Flatten(),
        )
        self.clf_mp1 = torch.nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Dropout(p = 0.2)
        )
        self.clf_mlp2 = nn.Sequential(
            nn.Linear(512,171)
        )

    def forward(self, x):
        x = self.resnet46(x)
        clf_fp1 = self.clf_layer1(x)
        clf_fp2 = self.clf_layer2(clf_fp1)
        clf_emb1 = self.clf_mp1(clf_fp2)
        clf_emb2 = self.clf_mlp2(clf_emb1)

        return clf_emb1,clf_emb2
# class ModelNet(nn.Module):
#     def __init__(self):
#         super(ModelNet, self).__init__()
#         self.resnet15 = BackboneResnet.resnet15()
#
#         self.clf_layer1 = torch.nn.Sequential(
#             BackboneResnet.BasicBlock(inplanes=256,planes=256),
#         )
#
#         self.clf_layer2 = torch.nn.Sequential(
#             BackboneResnet.BasicBlock(inplanes=256, planes=512,stride=2,downsample=torch.nn.Sequential(
#             BackboneResnet.conv1x1(256, 512 * BackboneResnet.BasicBlock.expansion, 2),
#             torch.nn.BatchNorm2d(512 * BackboneResnet.BasicBlock.expansion),
#         )),
#             BackboneResnet.BasicBlock(inplanes=512, planes=512),
#
#             torch.nn.AvgPool2d((7, 7), stride=1),
#             torch.nn.Flatten(),
#             torch.nn.Linear(512, 21),
#         )
#
#         self.ori_layer = nn.Sequential(
#             BackboneResnet.BasicBlock(inplanes=256, planes=256),
#
#             BackboneResnet.BasicBlock(inplanes=256, planes=512,stride=2,downsample=torch.nn.Sequential(
#             BackboneResnet.conv1x1(256, 512 * BackboneResnet.BasicBlock.expansion, 2),
#             torch.nn.BatchNorm2d(512 * BackboneResnet.BasicBlock.expansion),
#         )),
#             BackboneResnet.BasicBlock(inplanes=512, planes=512),
#
#             torch.nn.AvgPool2d((7, 7), stride=1),
#             torch.nn.Flatten(),
#             torch.nn.Linear(512, 19),
#         )
#
#     def forward(self, x):
#         x = self.resnet15(x)
#         clf_emb = self.clf_layer1(x)
#         clf_emb = self.clf_layer2(clf_emb)
#         return  clf_emb

