# -*- coding: utf-8 -*-
import h5py
import numpy as np


def get_images_labels(fl_path):
    group_name = 'images_labels'
    image_name = 'images'
    label_name = 'labels'
    with h5py.File(fl_path, 'r') as hr:
        images = hr[group_name + '/' + image_name][:]
        labels = hr[group_name + '/' + label_name][:]
    return images, labels

def get_gray_centre_data():
    flpath = "/home/hao/workspace/Sudoku Solver/My_Solver/printed_digits/arrays/gray_centre_my_own_data.h5"
    my_own_images, my_own_labels = get_images_labels(flpath)
    print my_own_images.shape,my_own_labels.shape

    flpath = "/home/hao/workspace/Sudoku Solver/My_Solver/printed_digits/arrays/gray_centre_online_data.h5"
    online_images, online_labels = get_images_labels(flpath)
    print online_images.shape, online_labels.shape
    return np.vstack([my_own_images,online_images]),np.vstack([my_own_labels,online_labels])

if __name__ =="__main__":
    get_gray_centre_my_own_data()
