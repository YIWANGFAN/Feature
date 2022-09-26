import torch.nn as nn
import torch
from torchvision import transforms, datasets
import numpy as np


import torch
import torch.nn as nn
class Extract(nn.Module):
    def __init__(self):
        super(Extract, self).__init__()
        model = torch.load('/home/daip/share/wxf/feature/network/resnet/dog_cat/resnet18_dog_cat_pre.pth')
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        out = self.avgpool(x)
        return out
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    # 加載模型
    model = Extract()
    model.to(device)
    model.eval()
    # print(model)
    from network.dataload import MyDataset

    # device : GPU or CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 数据转换
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "test": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
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
    batch_size = 1
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
    print(train_num, test_num)
    train_data = []
    train_lable = []
    train_save_path = '/home/daip/share/wxf/feature/feature/resnet/resnet18_dog_cat_train_pre.npz'
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        labels = labels-1
        labels = labels.to(device)
        outputs = model(images.to(device)).cpu()
        outputs = torch.squeeze(outputs)
        outputs = outputs.data.numpy()
        outputs = outputs.reshape(-1)
        labels = labels.cpu()
        labels = labels.data.numpy()
        labels = labels.reshape(-1)
        # print(outputs.shape)
        # print(labels)
        train_data.append(outputs)
        train_lable.append(labels)
        # lable.reshape(-1)
    train_lable = np.array(train_lable)
    train_lable = train_lable.reshape(-1)
    print(len(train_data))
    np.savez(train_save_path, vector=train_data, utt=train_lable)

    test_data = []
    test_lable = []
    test_save_path = '/home/daip/share/wxf/feature/feature/resnet/resnet18_dog_cat_test_pre.npz'
    for step, data in enumerate(test_loader, start=0):
        images, labels = data
        labels = labels-1
        labels = labels.to(device)
        outputs = model(images.to(device)).cpu()
        outputs = torch.squeeze(outputs)
        outputs = outputs.data.numpy()
        outputs = outputs.reshape(-1)
        labels = labels.cpu()
        labels = labels.data.numpy()
        labels = labels.reshape(-1)
        # print(len(outputs))
        # print(labels)
        test_data.append(outputs)
        test_lable.append(labels)
        # lable.reshape(-1)
    test_lable = np.array(test_lable)
    test_lable = test_lable.reshape(-1)
    print(len(test_data))
    np.savez(test_save_path, vector=test_data, utt=test_lable)

