import torch
from torch import nn
import BackboneResnet

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
            torch.nn.Linear(2048, 171),
        )

    def forward(self, x):
        x = self.resnet46(x)
        clf_emb = self.clf_layer1(x)
        clf_emb = self.clf_layer2(clf_emb)
        return clf_emb
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

