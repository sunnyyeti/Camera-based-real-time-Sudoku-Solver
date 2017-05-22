# -*- coding: utf-8 -*-
import cv2
import numpy as np


def get_size(img):
    """

    :param img:
    :return: return image size (width,height)
    """
    return img.shape[:2][::-1]


def add_rectangular(img, approx):
    """
    :param img:
    :param approx:
    :return:
    """
    for i in xrange(len(approx)):
        cv2.line(img,
                (approx[(i % 4)][0][0], approx[(i % 4)][0][1]),
                (approx[((i + 1) % 4)][0][0], approx[((i + 1) % 4)][0][1]),
                (255, 0, 0), 2)
    return img


def get_recs(ori_img):
    """

    :param ori_img: the input image
    :return: return a list of rectangulars based on the areas in descending order
    """
    w,h = get_size(ori_img)
    img_area = w*h
    thres = img_area/8.0
    bi_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    th2 = cv2.GaussianBlur(bi_img, (3, 3), 0, 0)
    kernel = np.ones((2, 2), np.uint8)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
    th2 = cv2.adaptiveThreshold(th2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                cv2.THRESH_BINARY_INV, 11, 0)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
    _, contours0, hierarchy = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    approximations = [cv2.approxPolyDP(ctr,15,True) for ctr in contours0]
    approximations_filter = [app for app in approximations if len(app)==4 and cv2.isContourConvex(app)]
    rectangles_area = [cv2.contourArea(app) for app in approximations_filter ]
    pair = zip(approximations_filter,rectangles_area)
    pair = [p for p in pair if p[1]>thres]
    pair.sort(key=lambda x:x[1],reverse=True)
    return pair



def get_rot_matrix(approximation, mapped_size, reverse = False):
    """

    :param approximation: the approxiamte rectangular
    :param mapped_size: mapped size (width, height) 
    :return: return a rotation matrix
    """
    ori_points = sort_apporximation(approximation)
    w, h = mapped_size
    map_points = np.array([ [0, 0],[w - 1, 0], [0, h - 1], [w - 1, h - 1]], dtype=np.float32)
    if reverse:
        rot_matrix = cv2.getPerspectiveTransform(map_points, ori_points )
    else:
        rot_matrix = cv2.getPerspectiveTransform(ori_points, map_points)
    return rot_matrix


def sort_apporximation(approximation):
    sum_xy = np.sum(approximation,axis=0)
    mean_y = float(sum_xy[0][1])/len(approximation)
    mean_x = float(sum_xy[0][0])/len(approximation)
    points = [x[0,:] for x in approximation]
    def locate(p):
        if p[1] < mean_y and p[0] < mean_x:
            return -1
        elif p[1] < mean_y and p[0] > mean_x:
            return 0
        elif p[1] > mean_y and p[0] < mean_x:
            return 1
        elif p[1] > mean_y and p[0] > mean_x:
            return 2
    points.sort(key=locate)
    return np.array(points,dtype=np.float32)


def add_v_h_grids(orig, width_split, height_split):
    """
    :param orig: the passed original picture
    :param height_split: the number of vertical split
    :param width_split: the number of horizontal split
    :return:
    """
    #print orig

    w, h = get_size(orig)
    hps = np.round(np.linspace(0,h-1,height_split+1)).astype(int)
    wps = np.round(np.linspace(0,w-1,width_split+1)).astype(int)
    for h_point in np.nditer(hps[1:-1]):
        cv2.line(orig,(0, h_point),(w-1, h_point),(255, 102, 255), 2)
    for w_point in np.nditer(wps[1:-1]):
        cv2.line(orig,(w_point,0),(w_point,h-1),(255,102,255),2)
    return orig


def get_valid_rectangulars(bin_img_block):
    """
    Assume the block is 28*28, specially for this task
    :param bin_img_block:
    :return:
    """
    _, contours0, hierarchy = cv2.findContours(bin_img_block, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(ctr) for ctr in contours0]
    valid_rects = [rect for rect in rects if rect[3] > rect[2] \
                   and 20 > rect[3] > 7 \
                   and 20 > rect[2] > 3
                   and 300 > rect[2] * rect[3] > 21 \
                   and 14 > rect[0] > 3 and 14 > rect[1] > 1]
    return valid_rects


def catch_digit_center(bin_img_block, digit_bound_size):
    """

    :param bin_img_block: the 28*28 binary image
    :param digit_bound_size: the size of the digit bound in the new picture
    :return: a 28*28 block with a digit in the center(if there is one)
    """
    rects = get_valid_rectangulars(bin_img_block)
    if len(rects)==0:
        return False, np.zeros((28,28))
    else:
        rects.sort(key=lambda x: x[2] * x[3],reverse=True)
        rec = rects[0]
        digit_bound = bin_img_block[rec[1]:rec[1]+rec[3]+1,rec[0]:rec[0]+rec[2]+1]
        if rec[3]*rec[2]<= digit_bound_size[0]*digit_bound_size[1]:
            digit = cv2.resize(digit_bound, digit_bound_size, interpolation=cv2.INTER_LINEAR)#Zooming
        else:
            digit = cv2.resize(digit_bound, digit_bound_size, interpolation=cv2.INTER_AREA)#shrinking
        left_top_x = int(np.round((28 - digit_bound_size[0])/2.0))
        left_top_y = int(np.round((28 - digit_bound_size[1])/2.0))
        digit_center = np.zeros((28,28))
        digit_center[left_top_y:left_top_y+digit_bound_size[1], left_top_x:left_top_x+digit_bound_size[0]] = digit
        return True, digit_center


def preprocess_sudoku_grid(mapped_pic):
    """

    :param mapped_pic: the  picture after doing warpPerspective for detecting digits
    :return: a binary picture
    """
    bi_img = cv2.cvtColor(mapped_pic, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((2, 2), np.uint8)
    th2 = cv2.adaptiveThreshold(bi_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 6)
    th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, kernel)
    return th2


def write_solution(mapped,digit_flag, answer):
    w,h = get_size(mapped)
    widthgap = w/9
    heightgap = h/9
    blank_flag = (1-digit_flag).astype(np.bool)
    for i in xrange(len(blank_flag)):
        if blank_flag[i]:
            hindex = i / 9
            windex = i % 9
            orig_point = (windex * widthgap, hindex * heightgap)
            cv2.putText(mapped, answer[i], (orig_point[0]+6, orig_point[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 1, 1), 2)
    return mapped

def split_2_blocks(orig, width_split, height_split):
    """

    :param orig: picture
    :param width_split: the number of horizontal split
    :param height_split: the number of vertical split
    :return:
    """
    points = []
    hor_splits = np.split(orig,height_split,axis=0)
    for h in hor_splits:
        for w in np.split(h,width_split,axis=1):
            points.append(w.flatten())
    return np.array(points)


def reflect_to_orig(orig, rot_matrix, mapped):
    """
    when we set the drawed pixel without zero RGB, then there is no problem.
    :param orig:
    :param rot_matrix:
    :param mapped:
    :return:
    """
    tmpori = orig.copy()
    tmpori = cv2.warpPerspective(mapped, rot_matrix, get_size(orig), tmpori, cv2.WARP_INVERSE_MAP)
    orig = orig*(tmpori == 0)
    merge = tmpori*(tmpori!=0)
    return orig + merge
