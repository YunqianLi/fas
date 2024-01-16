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

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

import datetime

import numpy as np

import logging
import sys


print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


torch.manual_seed(21)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# 设置超参数
BATCH_SIZE = 200  # 16
EPOCHS = 100  # 100
modellr = 1e-3
modellrs = []


# 数据
data_path_train = 'data/train2'
data_path_val = 'data/val2'


# 配置日志记录
logging.basicConfig(filename='./log/' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 重定向标准输出到日志文件
class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.rstrip():  # 防止空行被记录
            logging.log(self.level, message)

    def flush(self):
        pass


# 重定向标准输出和标准错误输出
sys.stdout = LoggerWriter(logging.INFO)
sys.stderr = LoggerWriter(logging.ERROR)

print('train3')
print('BATCH_SIZE:', BATCH_SIZE)
print('EPOCHS:', EPOCHS)
print('modellr:', modellr)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# setting device on GPU if available, else CPU
print('Using device:', DEVICE)
print()
# Additional Info when using cuda
if DEVICE.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')


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
dataset_train = myImageFolder(data_path_train, transform)
# print(dataset_train.imgs)
# 对应文件夹的label
print(dataset_train.class_to_idx)
dataset_test = myImageFolder(data_path_val, transform_test)
# 对应文件夹的label
print(dataset_test.class_to_idx)


# 导入数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)


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
    modellrnew = modellr * (0.1 ** (epoch // 30))
    print("lr:", modellrnew)
    for param_group in optimizer.param_groups:
        param_group['lr'] = modellrnew
    modellrs.append(modellrnew)


train_writer = SummaryWriter('./log/')
val_writer = SummaryWriter('./log/')

train_loss = []
train_acc = []
val_loss = []
val_acc = []


# 定义训练过程
def train(model, device, train_loader, optimizer, epoch):
    # 定义计算准确率的函数
    def compute_accuracy(outputs, labels):
        _, predicted = torch.max(outputs, 1)  # 获取每个样本中预测的类别
        correct = (predicted == labels).sum().item()  # 计算正确分类的样本数
        total = labels.size(0)  # 总样本数
        accuracy = correct / total  # 计算准确率
        return accuracy

    model.train()
    print()
    # Additional Info when using cuda
    if DEVICE.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
        print()

    sum_loss = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    # 在训练循环中计算准确率
    total_accuracy = 0.0
    total_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss

        accuracy = compute_accuracy(output, target)  # 计算批次的准确率
        total_accuracy += accuracy * target.size(0)  # 累加正确分类的样本数
        total_samples += target.size(0)  # 累加总样本数

        if (batch_idx + 1) % 50 == 0:
            print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))

    ave_loss = sum_loss / len(train_loader)
    final_accuracy = total_accuracy / total_samples
    print('epoch:{},loss:{}'.format(epoch, ave_loss))
    train_writer.add_scalar('Training Loss', ave_loss, global_step=epoch)
    print('epoch:{},Accuracy:{}'.format(epoch, final_accuracy))
    train_writer.add_scalar('Training Accuracy', final_accuracy, global_step=epoch)
    train_loss.append(ave_loss)
    train_acc.append(final_accuracy)


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

            # if (batch_idx + 1) % 50 == 0:
            #     print('{}\tValidation Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            #         epoch, (batch_idx + 1) * len(data), len(test_loader.dataset),
            #                100. * (batch_idx + 1) / len(test_loader), loss.item()))

        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))
        print('epoch:{}, validation loss:{}'.format(epoch, avgloss))
        print('epoch:{}, validation Accuracy:{}'.format(epoch, acc))
        print('Saving model in Epoch: {} '.format(epoch))
        torch.save(model, './model/model_%03d.pth' % epoch)
        val_writer.add_scalar('Val Loss', avgloss, global_step=epoch)
        val_writer.add_scalar('Val Accuracy', acc, global_step=epoch)
        val_loss.append(avgloss)
        val_acc.append(acc)


# 训练
if __name__ == "__main__":
    for epoch in range(1, EPOCHS + 1):
        print('=================================================')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'epoch', epoch, ' start')
        print('-------------------------------------------------')
        adjust_learning_rate(optimizer, epoch)
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' training...')
        train(model, DEVICE, train_loader, optimizer, epoch)
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' validation...')
        val(model, DEVICE, test_loader, epoch)
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'epoch', epoch, ' end')
        print('-------------------------------------------------')

        np.save('./model/train_loss.npy', np.array(train_loss))
        np.save('./model/train_acc.npy', np.array(train_acc))
        np.save('./model/val_loss.npy', np.array(val_loss))
        np.save('./model/val_acc.npy', np.array(val_acc))
        np.save('./model/modellrs.npy', np.array(modellrs))

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ' Done...')
