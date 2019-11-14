# -*- coding: utf-8 -*-
# @Time    : 2019/10/29 22:09
# @Author  : 0chen
# @FileName: bow.py
# @Software: PyCharm
# @Blog    : http://www.0chen.xyz

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import glob
from sklearn.cluster import KMeans
import pandas as pd

K_MEANS_EDN_CONDITION = 0.01
IMGSIZE = (300, 300)
K_CLUSTER = 1000
PICTURE_PATH = r'picture\*'
CENTERS_PATH = r'my_centers.csv'
PATH_FEATURE_PATH = r'my_path_feature.csv'
PICTURE_CLASS = ['bark', 'bike', 'boat', 'graf', 'leuven', 'tree', 'ubc', 'wall']
def get_all_picture_sift_descriptor(): # right
    all_descriptor = []
    img_path = glob.glob(PICTURE_PATH)
    for i, path in enumerate(img_path):
        print(path, i)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, IMGSIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        _, descriptors = sift.detectAndCompute(gray, None)
        all_descriptor.append(descriptors)
    return np.vstack(all_descriptor)

def get_picture_feature(k, centers):
    img_path = glob.glob(PICTURE_PATH)
    picture_feature = []
    for path in img_path:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, IMGSIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()
        _, descriptors = sift.detectAndCompute(gray, None)
        feature = np.zeros((k, ))
        for row in range(descriptors):
            index, _ = find_nearest_center(row, centers)
            feature[index] += 1
        picture_feature.append(feature)
    picture_feature = np.vstack(picture_feature)
    return [picture_feature, img_path]

def find_n_nearest(feature, imgpath, picture, n):
    imgpath_distance = dict()
    n = np.minimum(feature.shape[0], n)
    for row, path in zip(feature, imgpath):
        d = L2_norm(row, picture)
        imgpath_distance[path] = d
    ans = sorted(imgpath_distance.items(), key=lambda e: e[1])
    return [x[0] for x in ans][:n]

def show_picture(img_path, n_nearest_img_path):
    plt.figure()
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (100, 100))
    ax = plt.subplot2grid((3, 3), (0, 0))
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('origin img')
    for i, path in enumerate(n_nearest_img_path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (100, 100))
        ax = plt.subplot2grid((3, 3), (i//3+1, i%3))
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(str(i) +' nearest')
    plt.show()

def calculate_precision_recall(feature, imgpath, n):
    tp = 0
    for index in range(48):
        nearest = find_n_nearest(feature, imgpath, feature[index], n)
        for c in PICTURE_CLASS:
            if c in imgpath[index]:
                for img in nearest:
                    if c in img:
                        tp += 1
                break
    retrieved = 48 * 6
    relevant = 48 * 6

    precision = tp / retrieved
    recall = tp / relevant
    return [precision, recall]

def L1_norm(x, y): # right
    return np.linalg.norm(x - y, ord=1)

def L2_norm(x, y): # right
    return np.linalg.norm(x - y)

def distance(x, y): # right
    y = y.reshape(x.shape)
    return L1_norm(x, y)
    # return L2_norm(x, y)

def find_nearest_center(x, centers): # right
    index, min_d = 0, None
    for i, row in enumerate(centers):
        d = distance(x, row)
        if min_d is None or d < min_d:
            index, min_d = i, d
    return [index, min_d]

def centers_distance(centers1, centers2): # right
    return np.sqrt(np.power(centers1-centers2, 2).sum(axis=1)).sum()


def k_means(data, k):
    x, y = data.shape
    min_all_cost = None
    for _ in range(1):  # K means 循环次数，防止随机到意外情况
        centers = data[random.sample(range(x), k)]
        # 如果 centers 选择到了相同的点，会出错，但是点为什么会相同? 除非sift选择的特征会相同。
        # 那就假定sift的特征选择不会有相同吧，感觉概率也非常低
        while True:
            classify = [[] for _ in range(k)]
            allcost = 0
            for row in data:
                index, cost = find_nearest_center(row, centers)
                classify[index].append(row)
                allcost += cost

            new_centers = []
            for i in range(k):
                classify[i] = np.vstack(classify[i])  # 如果sift特征相同，并且刚好选到相同特征，会出错
                new_centers.append(classify[i].mean(axis=0))
            new_centers = np.vstack(new_centers)
            distance = centers_distance(centers, new_centers)
            if distance < K_MEANS_EDN_CONDITION:
                break

            centers = new_centers
            print("k = {}, data_length = {}, centers_distance={}".format(k, x, distance))

        if min_all_cost is None or min_all_cost > allcost:
            min_all_cost = allcost
    return new_centers





if __name__ == '__main__':
    # all_descriptor = get_all_picture_sift_descriptor() # openCV 找所有图片特征不变点
    # print(all_descriptor.shape) # 输出特征不变点总个数
    #
    # # centers = k_means(all_descriptor, K_CLUSTER)   # 自己写的K means 对 SFIT 选择的特征值聚类
    #
    # kmeans = KMeans(n_clusters=K_CLUSTER, random_state=0).fit(all_descriptor)  # scikit 的K means 对 SFIT 选择的特征值聚类
    # centers = kmeans.cluster_centers_
    #
    # centers = pd.DataFrame(centers)                           # 备份 聚类中心
    # centers.to_csv(CENTERS_PATH, index=False, header=False)
    #
    # features, img_path = get_picture_feature(K_CLUSTER, centers)   # 计算张图片的特征
    #
    # path_feature = pd.DataFrame(data=features, index=img_path)    # 备份图片的特征
    # path_feature.to_csv(PATH_FEATURE_PATH, header=False)
    #

    centers = pd.read_csv(CENTERS_PATH, header=None)       # 读取备份的聚类中心

    path_feature = pd.read_csv(PATH_FEATURE_PATH, index_col=0, header=None)  # 读取备份的图片特征
    imgpath = path_feature.index
    feature = path_feature.values


    search_index = np.random.randint(48)  # 随机一张图片，用来检索

    N = 6
    nearest = find_n_nearest(feature, imgpath, feature[search_index], N) # 寻找最近的 N=6 张图片
    show_picture(imgpath[search_index], nearest) # 将找到的 图片按最接近顺序排列

    precision, recall = calculate_precision_recall(feature, imgpath, N)
    print('precision = {}, recall = {}'.format(precision, recall))