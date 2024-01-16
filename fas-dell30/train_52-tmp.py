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

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import datetime

import numpy as np

import time

# 设置超参数
BATCH_SIZE = 32  # 16
EPOCHS = 400  # 100
modellr = 1e-4  # 1e-4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
# 读取数据
dataset_train = myImageFolder('data/train', transform)
dataset_test = myImageFolder('data/val', transform_test)
# 导入数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False)


# 实例化模型并且移动到GPU
model = torchvision.models.resnet18(pretrained=True)
# 修改输入输出层
model.conv1 = nn.Conv2d(52, 64, kernel_size=7, stride=2, padding=3, bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.to(DEVICE)


# 损失函数
criterion = nn.CrossEntropyLoss()


# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model.parameters(), lr=modellr)

# 创建GPU事件
start_event_step1 = torch.cuda.Event(enable_timing=True)
end_event_step1 = torch.cuda.Event(enable_timing=True)
start_event_step2 = torch.cuda.Event(enable_timing=True)
end_event_step2 = torch.cuda.Event(enable_timing=True)


# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    start_event_step1.record()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print('batch_idx:', batch_idx)
        data, target = Variable(data).to(device), Variable(target).to(device)

        # start_event_step2.record()

        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # end_event_step2.record()
        # torch.cuda.synchronize()
        # print('t2:{:.2f}'.format(start_event_step2.elapsed_time(end_event_step2)))

        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 10 == 0:
            print('Time:{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch,
                (batch_idx + 1) * len(data),
                len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                loss.item()))

        if batch_idx == 100:
            break

    end_event_step1.record()
    torch.cuda.synchronize()
    print('t1:{:.2f}'.format(start_event_step1.elapsed_time(end_event_step1)))

    ave_loss = sum_loss / len(train_loader)
    print('[TRAIN] epoch:{},loss:{}'.format(epoch, ave_loss))


# 定义验证过程
def val(model, device, test_loader, epoch):
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
        print('[VAL] epoch:{},loss:{}'.format(epoch, avgloss))


# main
if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        val(model, DEVICE, test_loader, epoch)
