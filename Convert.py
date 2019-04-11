# coding=utf-8
#将原始图片缩放到100 x 100，并变换成numpy数组(shape=[100,100,3])，做成数据集
#数据集的（见xxxArrayData.dat)的结构：
# 使用pickle.load可以加载后，为一个shape = [num, 100, 100, 3]的四维数组，num为数据集中的图片数，之后3维为图片数据
#本程序使用方法
# python Convert.py 
#  对文件夹下所有图片文件夹做成ArrayData数据集
# python Convert.py Reimu
#  对Reimu文件夹中的博丽灵梦的图片做成ArrayData数据集，存放到ReimuArrayData/dataset.dat中

import sys
try:
    from PIL import Image
except:
    if __name__ == "__main__":
        puts("This program requires python3 PIL library, please pip3 install PIL")
        sys.exit(0)
    else:
        pass
import os
import shutil
import pickle
import numpy as np

BASE_DIR = os.path.split(os.path.realpath(__file__))[0]

CLASSES = ["Reimu", "Marisa", "Koishi", "Remilia", "Sakuya", "Cirno", "Clownpiece", "Flandre", "Patchouli",
           "Yuyuko", "Hina", "Kaguya", "Yukari", "Yuuka", "Eirin", "Kogasa", "Youmu", "Sanae",
           "Utsuho", "Satori"]
CLASSES_NAME = ["博丽灵梦", "雾雨魔理沙", "古明地恋", "蕾米莉亚斯卡雷特", "十六夜咲夜", "琪露诺", "克劳恩皮丝", "芙兰朵露斯卡雷特", "帕秋莉诺雷姬",
                "西行寺幽幽子", "键山雏", "蓬莱山辉夜", "八云紫", "风见幽香", "八意永琳", "多多良小伞", "魂魄妖梦", "东风谷早苗",
                "灵乌路空", "古名地觉"]

OUTPUT_SIZE = 100    #输出100 x 100
def extract_image(img):
    if isinstance(img, str):
        try:
            img = Image.open(img, "r")
        except:
            return [[[]]] #返回空数据
    img = img.resize(size=(OUTPUT_SIZE, OUTPUT_SIZE), resample=Image.BILINEAR)
    pix = img.load()
    w = img.size[0]
    h = img.size[1]
    data = []
    for i in range(h):
        row = []
        for j in range(w):
            try:
                c = pix[i, j]
                row.append([c[0] / 255.0, c[1] / 255.0, c[2] / 255.0])  #归一化
            except:
                if isinstance(img, str):
                    print("Error when extracting {}, maybe unsupported format".format(img))
                return [[[]]]
        data.append(row)
    return data

def generate_train_data(filename):
    img = Image.open(filename, "r")
    return [extract_image(img),  #原始图像
            extract_image(img.transpose(Image.FLIP_LEFT_RIGHT)),  #左右翻转
            extract_image(img.transpose(Image.FLIP_TOP_BOTTOM))]  #上下翻转

def save_array_data(input_dir, output_dir):
    if not os.path.exists(input_dir):
        print("Directory {} does not exist, skipped.".format(input_dir))
        return 
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    dataset = []
    for _fn in os.listdir(input_dir):
        d = generate_train_data(os.path.join(input_dir, _fn))
        for i in range(3):
            if len(d[i]) == OUTPUT_SIZE:
                dataset.append(d[i])
    with open(os.path.join(output_dir, "dataset.dat"), "wb") as f:
        pickle.dump(dataset, f)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        for key in CLASSES:
            input_dir = os.path.join(BASE_DIR, key)
            output_dir = os.path.join(BASE_DIR, key + "ArrayData")
            save_array_data(input_dir, output_dir)
    else:
        input_dir = os.path.join(BASE_DIR, sys.argv[1])
        output_dir = os.path.join(BASE_DIR, sys.argv[1] + "ArrayData")
        save_array_data(input_dir, output_dir)