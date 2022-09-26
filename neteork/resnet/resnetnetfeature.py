import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets



import torch
import torch.nn as nn

class Extract(nn.Module):
    def __init__(self):
        super(Extract, self).__init__()
        model = torch.load('/home/daip/share/old_share/wxf/feature/network/resnet/car/resnet50_2022_8_9_car.pth')
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

model = Extract()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


transforms = transforms.Compose([
    transforms.Resize((224,224)),  # 将图片短边缩放至256，长宽比保持不变：
    transforms.ToTensor()  # 把图片进行归一化，并把数据转换成Tensor类型
])
data_train = datasets.ImageFolder('/home/daip/share/old_share/wxf/datasets/car/train', transform=transforms)
data_test = datasets.ImageFolder('/home/daip/share/old_share/wxf/datasets/car/val',transform=transforms)
# 加载数据集
dataTrainLoader = torch.utils.data.DataLoader(data_train, batch_size=1, shuffle=True)
dataTestLoader = torch.utils.data.DataLoader(data_test, batch_size=1)
dog_test_data = []
dog_test_lable = []
for step, data in enumerate(dataTestLoader, start=0):
    images, labels = data
    labels = labels.to(device)
    outputs = model(images.to(device)).cpu()
    outputs = torch.squeeze(outputs)
    outputs = outputs.data.numpy()
    outputs = outputs.reshape(-1)
    labels = labels.cpu()
    labels = labels.data.numpy()
    labels = labels.reshape(-1)
    print(outputs.shape)
    print(labels)
    dog_test_data.append(outputs)
    dog_test_lable.append(labels)
    # lable.reshape(-1)
dog_lable = np.array(dog_test_lable)
dog_lable = dog_lable.reshape(-1)
print(len(dog_test_data))
np.savez('/home/daip/share/old_share/wxf/feature/feature/car/resnet/car_test_res.npz', vector=dog_test_data, utt=dog_lable)



