
from torch.autograd import Variable
from network.dataload_tea import MyDataset

from torch.utils.data import DataLoader




import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import  models, transforms


import time



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root = '/home/daip/share/old_share/wxf/datasets/tea-cake/'
traindata = MyDataset(txt=root + 'teatrain.txt', transform=transforms.ToTensor())
train_data = DataLoader(traindata,batch_size=32,shuffle=True)
test_data = MyDataset(txt=root + 'teatest.txt', transform=transforms.ToTensor())
testdata = DataLoader(test_data,batch_size=32,shuffle=True)
trainsize = len(traindata)
test_size = len(test_data)
model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = True
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train(model,criterion,optimizer, scheduler, num_epochs=100):
    best_acc = 0
    since = time.time();
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs,labels in train_data:
            inputs = inputs.to(device)
            labels = labels.to(device)


            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        scheduler.step()
        epoch_loss = running_loss / trainsize
        epoch_acc = running_corrects.double() / trainsize
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            "train", epoch_loss, epoch_acc))
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        model.eval()  # 测试过程中不需要dropout，使用所有的神经元
        acc_test = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for test_data in testdata:
                test_images, test_labels = test_data
                test_labels = test_labels
                outputs = model(test_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc_test += (predict_y == test_labels.to(device)).sum().item()
                # print(acc_test)
            test_accurate = acc_test / test_size

            if test_accurate > best_acc:
                best_acc = test_accurate
                torch.save(model, '/home/daip/share/old_share/wxf/feature/network/resnet/tea-cake/resnet50_tea.pth')
            print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1, epoch_loss, test_accurate))
train(model, criterion, optimizer, exp_lr_scheduler,
                       num_epochs=10)