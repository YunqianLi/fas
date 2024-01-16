
# 代码参考链接：https://juejin.cn/post/7012922120392933383

import torch.optim as optim
import torch
import torch.nn as nn
# import torch.nn.parallel
# import torch.optim
# import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models
# from effnetv2 import effnetv2_s
from torch.autograd import Variable

from lib.loadTifImage import myImageFolder

import datetime
import datetime
# time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
# print(time)

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# 设置超参数

BATCH_SIZE = 16
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# setting device on GPU if available, else CPU
print('Using device:', DEVICE)
print()
#Additional Info when using cuda
if DEVICE.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')


# 数据预处理

transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomCrop(50),
    # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    # transforms.Normalize()
])
transform_test = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize()
])
# 读取数据
# dataset_train = datasets.ImageFolder('data/train', transform)
dataset_train = myImageFolder('data/train', transform)
# print(dataset_train.imgs)
# 对应文件夹的label
print(dataset_train.class_to_idx)
dataset_test = myImageFolder('data/val', transform_test)
# 对应文件夹的label
print(dataset_test.class_to_idx)

# 导入数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)
modellr = 1e-4

# 实例化模型并且移动到GPU
criterion = nn.CrossEntropyLoss()
# model = effnetv2_s()
# num_ftrs = model.classifier.in_features
# model.classifier = nn.Linear(num_ftrs, 2)
model = torchvision.models.resnet18(pretrained=True)
model.conv1 = nn.Conv2d(52, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.to(DEVICE)
# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model.parameters(), lr=modellr)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    modellrnew = modellr * (0.1 ** (epoch // 50))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew


# 定义训练过程

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    print()
    # Additional Info when using cuda
    if DEVICE.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))

def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = Variable(data).to(device), Variable(target).to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))
        print('Saving model in Epoch: {} '.format(epoch))
        torch.save(model, './model/model_%03d.pth' % epoch)


# 训练
if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        adjust_learning_rate(optimizer, epoch)
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        train(model, DEVICE, train_loader, optimizer, epoch)
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        val(model, DEVICE, test_loader)
    print('Saving model......')
    torch.save(model, 'model.pth')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
