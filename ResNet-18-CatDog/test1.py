import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.autograd import Variable
import os
from PIL import Image

# classes = ('cat', 'dog')
classes = ('fake', 'true')

transform_test = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth")
model.eval()
model.to(DEVICE)
path = 'data/test1_fake/'
testList = os.listdir(path)
for file in testList:
    img = Image.open(path + file)
    img = transform_test(img)
    img.unsqueeze_(0)
    img = Variable(img).to(DEVICE)
    out = model(img)
    # Predict
    _, pred = torch.max(out.data, 1)
    prob = torch.nn.functional.softmax(out.data, 1)
    print(
        'Image Name:{},predict:{},confidence:{},probability:{}'.format(file, classes[pred.data.item()], out.data, prob))
