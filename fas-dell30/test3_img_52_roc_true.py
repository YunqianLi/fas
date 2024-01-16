import torch.utils.data.distributed
import torchvision.transforms as transforms

from torch.autograd import Variable
import os
from PIL import Image

import numpy as np

# 测试数据路径
path = 'data/test/TEST/TRUE/'
model_num = '057'

# classes = ('cat', 'dog')
classes = ('fake', 'true')

transform_test = transforms.Compose([
    # transforms.Resize((64, 64)),
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 加载模型
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('./model/model_' + model_num + '.pth')
model.eval()
model.to(DEVICE)

# 读取基的坐标
img_D_arr = np.array(Image.open('data/test/BASE/D/D_1.tif'))
patch_size = 64
patch_radius = int(patch_size / 2)

# 读取51个基
img_B_arr = np.zeros((5120, 5120, 51), np.uint8)
img_B_path = 'data/test/BASE/'
img_B_list = [ '1.tif', '10.tif', '11.tif', '12.tif', '13.tif', '14.tif', '15.tif', '16.tif', '17.tif', '18.tif', '19.tif',
               '2.tif', '20.tif', '21.tif', '22.tif', '23.tif', '24.tif', '25.tif', '26.tif', '27.tif', '28.tif', '29.tif',
               '3.tif', '30.tif', '31.tif', '32.tif', '33.tif', '34.tif', '35.tif', '36.tif', '37.tif', '38.tif', '39.tif',
               '4.tif', '40.tif', '41.tif', '42.tif', '43.tif', '44.tif', '45.tif', '46.tif', '47.tif', '48.tif', '49.tif',
               '5.tif', '50.tif', '51.tif', '6.tif', '7.tif', '8.tif', '9.tif' ]
for num, file in enumerate(img_B_list):
    print('loading base image {}'.format(file))
    img_B_arr[:, :, num] = np.array(Image.open(img_B_path + file))

# 读取测试图像
testList = [f for f in os.listdir(path) if '.tif' in f]

# 读取GT图像
GTList = [f for f in os.listdir(path + '/GT/') if '.png' in f]

# 遍历图像
face_counter = 0
face_counter_true = 0
for idx, file in enumerate(testList):
    face_counter += 1
    filename_prefix = os.path.splitext(file)
    filename_GT = filename_prefix[0] + '.png'
    print('loading GT image : {}, {}'.format(face_counter, filename_GT))
    img_gt = Image.open(path + '/GT/' + filename_GT)
    img_gt_arr = np.array(img_gt)
    print('processing image : {}, {}'.format(face_counter, file))
    img_big = Image.open(path + file)
    # img_big.show()
    img_big_arr = np.array(img_big)  # .T
    # img_big_t = Image.fromarray(img_big_arr)
    # img_patch_arr = np.zeros((patch_size, patch_size, 3), np.uint8)
    img_patch_arr = np.zeros((patch_size, patch_size, 52), np.uint8)
    img_out_arr = np.zeros_like(img_D_arr)
    # img_big_t.show()

    height, width = img_D_arr.shape
    patch_counter = 0  # 9275
    patch_counter_true = 0
    patch_counter_2 = 9275
    probs = np.zeros(patch_counter_2)
    for h in range(patch_size, height - patch_size):
        for w in range(patch_size, width - patch_size):
            if (img_D_arr[h, w] == 255) & (img_gt_arr[w, h] == 255):
                # print('patch_counter = {}, (h,w) = ({},{})'.format(patch_counter, h, w))
                left = w - patch_radius
                top = h - patch_radius
                right = w + patch_radius
                down = h + patch_radius

                img_patch_arr[:, :, 0] = img_big_arr[top:down, left:right]
                img_patch_arr[:, :, 1:52] = img_B_arr[top:down, left:right]
                #
                # img_patch_arr[:, :, 0] = img_big_arr[top:down, left:right]
                # img_patch_arr[:, :, 1] = img_big_arr[top:down, left:right]
                # img_patch_arr[:, :, 2] = img_big_arr[top:down, left:right]
                # img_patch = img_big_t.crop([left, top, right, down])

                # print('(left, top, right, down) = ({},{},{},{}, img_patch.shape() = {})'.format(left, top, right, down,
                #                                                                                 img_patch_arr.shape))

                # img_patch = Image.fromarray(img_patch_arr)
                # img = transform_test(img_patch)
                img = transform_test(img_patch_arr)
                img.unsqueeze_(0)
                img = Variable(img).to(DEVICE)
                out = model(img)
                # Predict
                _, pred = torch.max(out.data, 1)
                prob = torch.nn.functional.softmax(out.data, 1)
                # print('Image Name:{},predict:{},confidence:{},probability:{}'.format(file, classes[pred.data.item()],
                #                                                                      out.data, prob))
                probs[patch_counter] = prob[0, 1].data.item()
                patch_counter += 1

                if pred.data.item() == 1:  # true face
                    img_out_arr[top:down, left:right] = 255
                    patch_counter_true += 1
                    # print('true patch counter = {}'.format(patch_counter_true))

    ratio_true = (float(patch_counter_true) / float(patch_counter))
    if ratio_true > 0.1:
        face_counter_true += 1
    print('face_counter_true / face_counter = {}/{}, ratio_true = {}'.format(patch_counter_true, patch_counter,
                                                                             ratio_true))

    prefix = file[:-4] + '_' + model_num + '_' + format(ratio_true * 100, '.2f') + '_' + str(patch_counter)

    img_out = Image.fromarray(img_out_arr)
    img_out.save(path + prefix + '.png')  # save png
    print('saving image : {}, {}'.format(face_counter, prefix + '.png'))

    probs2 = probs[0:patch_counter]
    np.save(path + prefix + '.npy', probs2)  # save probs
    print(probs2)
    print('\n')
