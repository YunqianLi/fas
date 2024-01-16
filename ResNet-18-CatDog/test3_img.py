import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.autograd import Variable
import os
from PIL import Image

import numpy as np

# classes = ('cat', 'dog')
classes = ('fake', 'true')

transform_test = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model.pth")
model.eval()
model.to(DEVICE)
path = 'data/test/true/'

img_D_arr = np.array(Image.open('data/test/D/D_1.tif'))
patch_size = 64
patch_radius = int(patch_size / 2)

testList = [f for f in os.listdir(path) if '.tif' in f]
face_counter = 0
face_counter_true = 0
for file in testList:
    face_counter += 1
    print('processing image : {}, {}'.format(face_counter, file))
    img_big = Image.open(path + file)
    # img_big.show()
    img_big_arr = np.array(img_big) # .T
    # img_big_t = Image.fromarray(img_big_arr)
    img_patch_arr = np.zeros((patch_size, patch_size, 3), np.uint8)
    img_out_arr = np.zeros_like(img_D_arr)
    # img_big_t.show()

    height, width = img_D_arr.shape
    patch_counter = 0
    patch_counter_true = 0
    for h in range(patch_size, height - patch_size):
        for w in range(patch_size, width - patch_size):
            if img_D_arr[h, w] == 255:
                patch_counter += 1
                # print('patch_counter = {}, (h,w) = ({},{})'.format(patch_counter, h, w))
                left = w - patch_radius
                top = h - patch_radius
                right = w + patch_radius
                down = h + patch_radius
                img_patch_arr[:, :, 0] = img_big_arr[top:down, left:right]
                img_patch_arr[:, :, 1] = img_big_arr[top:down, left:right]
                img_patch_arr[:, :, 2] = img_big_arr[top:down, left:right]
                # img_patch = img_big_t.crop([left, top, right, down])

                # print('(left, top, right, down) = ({},{},{},{}, img_patch.shape() = {})'.format(left, top, right, down,
                #                                                                                 img_patch_arr.shape))

                img_patch = Image.fromarray(img_patch_arr)
                img = transform_test(img_patch)
                img.unsqueeze_(0)
                img = Variable(img).to(DEVICE)
                out = model(img)
                # Predict
                _, pred = torch.max(out.data, 1)
                prob = torch.nn.functional.softmax(out.data, 1)
                # print('Image Name:{},predict:{},confidence:{},probability:{}'.format(file, classes[pred.data.item()],
                #                                                                      out.data, prob))

                if pred.data.item() == 1:  # true face
                    img_out_arr[top:down, left:right] = 255
                    patch_counter_true += 1
                    # print('true patch counter = {}'.format(patch_counter_true))

    ratio_true = (float(patch_counter_true) / float(patch_counter))
    if ratio_true > 0.1:
        face_counter_true += 1
    print('face_counter_true / face_counter = {}/{}, ratio_true = {}'.format(patch_counter_true, patch_counter, ratio_true))

    img_out = Image.fromarray(img_out_arr)
    img_out.save(path + file[:-4] + '.png')
    print('saving image : {}, {}'.format(face_counter, file[:-4] + '.png'))
    print('\n')
