import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms, datasets



import torch
import torch.nn as nn

class Extract(nn.Module):
    def __init__(self):
        super(Extract, self).__init__()
        model = torch.load('/home/daip/share/old_share/wxf/feature/network/vgg/car/vgg16_2022_8_9_false_car.pth')
        self.features = model.features
        self.avgpool = model.avgpool

        fc1 = nn.Linear(25088, 4096)
        fc1.bias = model.classifier[0].bias
        fc1.weight = model.classifier[0].weight

        fc2 = nn.Linear(4096, 4096)
        fc2.bias = model.classifier[3].bias
        fc2.weight = model.classifier[3].weight

        self.classifier = nn.Sequential(
            fc1,
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            fc2,
            )


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

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
for step, data in enumerate(dataTrainLoader, start=0):
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
np.savez('/home/daip/share/old_share/wxf/feature/feature/car/vgg/car_train_vgg.npz', vector=dog_test_data, utt=dog_lable)



