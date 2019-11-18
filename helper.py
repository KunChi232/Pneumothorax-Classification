from PIL import Image
from glob import glob
import os
import json, random
import numpy as np

normal_path = 'Normal/normal/'
abnormal_path = 'Postive_pure/1 pure pneumothorax/'




def get_data():
    normal_img = np.asarray(load_images_from_folder(normal_path))
    normal_label = np.zeros(len(normal_img))
    normal_label = [[1,0] if normal_label[i] == 0 else [0,1] for i in range(len(normal_label))]
    abnormal_img = np.asarray(load_images_from_folder(abnormal_path))
    abnormal_label = np.ones(len(abnormal_img))
    abnormal_label = [[1,0] if abnormal_label[i] == 0 else [0,1] for i in range(len(abnormal_label))]
    imgs = np.concatenate((normal_img, abnormal_img))
    labels = np.concatenate((normal_label, abnormal_label))
    imgs_new, labels_new = shuffle(imgs, labels)

    # img = Image.fromarray(imgs_new[1])
    # img.show()
    # print(imgs_new[0])
    # print(labels_new)
    return imgs_new[:90], labels_new[:90], imgs_new[90:], labels_new[90:]

def shuffle(data, label):
    p = np.random.permutation(len(data))
    return data[p], label[p]

def get_abnormal_label(folder):
    labels = list()
    for filename in glob(folder+'*.json'):
        js = open(filename)
        js = json.load(js)
        print(js)

def load_images_from_folder(folder):
    images = []
    for filename in glob(folder + '*.png'):
        print(filename)
        img = Image.open(filename)
        img = img.resize((224,224))
        img = np.asarray(img)
        images.append(img)
    return images