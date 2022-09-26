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
import os
import json
import time
from torch.optim.lr_scheduler import StepLR
from model import getmodel


#device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#数据转换
# 数据增强  训练与测试区域别
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 ]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               ])}

train_dataset = datasets.ImageFolder(root='/home/daip/share/old_share/wxf/datasets/car/train',
                                     transform=data_transform["train"])
train_num = len(train_dataset)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root='/home/daip/share/old_share/wxf/datasets/car/val',
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=0)





modelname = 'vgg16'
num_class = 10
net = getmodel(modelname,num_class,False)

net.to(device)

#损失函数:这里用交叉熵
loss_function = nn.CrossEntropyLoss()
#优化器 这里用Adam
optimizer = optim.SGD(net.parameters(), lr=0.001,momentum=0.9,weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
#训练参数保存路径

save_path = '/home/daip/share/old_share/wxf/feature/network/vgg/car/vgg16_2022_8_9_false_car.pth'
#训练过程中最高准确率
best_acc = 0.0

#开始进行训练和测试，训练一轮，测试一轮
for epoch in range(50):
    # train
    net.train()    #训练过程中，使用之前定义网络中的dropout
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        # print statistics
        running_loss += loss.item()
        # print train process
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
    print()
    print(time.perf_counter()-t1)

    # validate
    net.eval()    #测试过程中不需要dropout，使用所有的神经元
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            acc += (predict_y == val_labels.to(device)).sum().item()
        val_accurate = acc / val_num
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net, save_path)
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / step, val_accurate))

print('Finished Training')

