# -*- coding: utf-8 -*-
# @Time    : 2021/9/10 9:48 上午
# @Author  : wxf
# @FileName: train.py
# @Software: PyCharm
# @Email ：15735952634@163.com
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from network.densenet.pre_model import model
from network.densenet.no_pre_model import densenet121,densenet161,densenet169,densenet201
import os
import json
import time
from network.dataload_tea import MyDataset


#device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#数据转换
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 ]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "test":transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor()
                               ])}

# train_dataset = datasets.ImageFolder(root='/home/daip/PycharmProjects/wxf/FeatureEx/data/tea_data/train',
#                                      transform=data_transform["train"])

#
root = '/home/daip/share/old_share/wxf/datasets/tea-cake/'
batch_size = 64
train_dataset = MyDataset(txt=root + 'teatrain.txt', transform=data_transform["train"])
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
test_dataset = MyDataset(txt=root + 'teatest.txt', transform=data_transform["test"])
test_loader = torch.utils.data.DataLoader(test_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
train_num = len(train_dataset)
test_num = len(test_dataset)
print(train_num,test_num)

# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()
#print(test_image[0].size(),type(test_image[0]))
#print(test_label[0],test_label[0].item(),type(test_label[0]))

def train(is_pre,modelname):

    num_class = 37
    if is_pre == True:
        save_path = './tea-cake/'+modelname+'_tea_pre.pth'

        epochs = 20
        if modelname == 'densenet121':
            net = model(modelname,num_class)
        if modelname == 'densenet161':
            net = model(modelname,num_class)
        if modelname == 'densenet169':
            net = model(modelname,num_class)
        if modelname == 'densenet201':
            net = model(modelname,num_class)
    else:
        save_path = './dog_cat/' + modelname + '_dog_cat_no_pre.pth'

        epochs = 50
        if modelname == 'densenet121':
            net = densenet121(num_classes=num_class)
        if modelname == 'densenet161':
            net = densenet161(num_classes=num_class)
        if modelname == 'densenet169':
            net = densenet169(num_classes=num_class)
        if modelname == 'densenet201':
            net = densenet201(num_classes=num_class)

    net.to(device)

    #损失函数:这里用交叉熵
    loss_function = nn.CrossEntropyLoss()
    #优化器 这里用Adam
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    #训练参数保存路径


    #训练过程中最高准确率
    best_acc = 0.0
    print(save_path)
    #开始进行训练和测试，训练一轮，测试一轮
    for epoch in range(epochs):
        # train
        net.train()    #训练过程中，使用之前定义网络中的dropout
        running_loss = 0.0
        t1 = time.perf_counter()
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            labels = labels
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()
        print(time.perf_counter()-t1)

        net.eval()  # 测试过程中不需要dropout，使用所有的神经元
        acc_test = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for test_data in test_loader:
                test_images, test_labels = test_data
                test_labels = test_labels-1
                outputs = net(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc_test += (predict_y == test_labels.to(device)).sum().item()
            test_accurate = acc_test / test_num
            if test_accurate > best_acc:
                best_acc = test_accurate
                torch.save(net, save_path)
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1, running_loss / step, test_accurate))


    print('Finished Training')
    return best_acc
if __name__ == '__main__':
    is_pres = [True]
    modelnames = ['densenet121']
    acc = []
    for is_pre in is_pres:
        for modelname in modelnames:
            print(is_pre,modelname)
            best_acc = train(is_pre,modelname)
            acc.append(best_acc)
    print(acc)
