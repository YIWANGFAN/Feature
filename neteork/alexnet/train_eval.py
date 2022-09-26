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
from model import BuildAlexNet
import os
import json
import time
from network.dataload import MyDataset


#device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


#数据转换
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "test":transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor()
                               ])}

# train_dataset = datasets.ImageFolder(root='/home/daip/PycharmProjects/wxf/FeatureEx/data/tea_data/train',
#                                      transform=data_transform["train"])

#
root = '/home/daip/share/wxf/feature/data/annotations/'
all_dataset = MyDataset(txt=root + 'alldata.txt', transform=data_transform["val"])
train_size = int(0.8 * len(all_dataset))
test_size = len(all_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
train_num = len(train_dataset)
test_num = len(test_dataset)
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

#
# validate_dataset = datasets.ImageFolder(root='/home/daip/PycharmProjects/wxf/FeatureEx/data/tea_data/test',
#                                         transform=data_transform["val"])

# validate_dataset = MyDataset(txt=root + 'valid.txt', transform=data_transform["val"])
# val_num = len(validate_dataset)
# validate_loader = torch.utils.data.DataLoader(validate_dataset,
#                                               batch_size=batch_size, shuffle=True,
#                                               num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=0)
print(train_num,test_num)

# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()
#print(test_image[0].size(),type(test_image[0]))
#print(test_label[0],test_label[0].item(),type(test_label[0]))





net = BuildAlexNet(model_type='new',n_output=37)
print(net)
net.to(device)

#损失函数:这里用交叉熵
loss_function = nn.CrossEntropyLoss()
#优化器 这里用Adam
optimizer = optim.Adam(net.parameters(), lr=0.0002)
#训练参数保存路径

save_path = './dog_cat/AlexNet_dog_cat_no_pre.pth'
#训练过程中最高准确率
best_acc = 0.0

#开始进行训练和测试，训练一轮，测试一轮
for epoch in range(100):
    # train
    net.train()    #训练过程中，使用之前定义网络中的dropout
    running_loss = 0.0
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        labels = labels-1
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


