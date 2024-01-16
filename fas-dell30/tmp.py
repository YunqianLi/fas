import os

def mysort(filename):
    filename2 = os.path.splitext(filename)
    num2 = int(filename2[0])
    return num2

img_B_path = 'data/test/BASE/'
img_B_list = [f for f in os.listdir(img_B_path) if '.tif' in f]
sorted_files = sorted(img_B_list, key=mysort)
print(sorted_files)