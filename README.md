# Image-search(Bow)
本次实验为实验Bow算法实现图片检索,以下为实验结果
### 数据集
来自 [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/data/affine/ "VGG")，一共48张图片，分成8个类别，每个类别 6 张图

### 实验步骤
 1. 通过 SIFT 算法计算 48 张图的特征不变点
 2. 用 K-means 算法对 SIFT 算法计算得到特征不变点聚类
 3. 计算每张图片的特征值
 4. 输入一张图片，得到最相近的 6 张图片
 5. 计算 precision 和 recall

### 图片检索结果
- 结果好的
[![](https://i.loli.net/2019/10/31/ohB3aF1PUYl7f2H.png)](https://i.loli.net/2019/10/31/ohB3aF1PUYl7f2H.png)

- 结果中等的
[![](https://i.loli.net/2019/10/31/voLej2aJwEAFX4W.png)](https://i.loli.net/2019/10/31/voLej2aJwEAFX4W.png)

- 结果不好的
[![](https://i.loli.net/2019/10/31/oUCMX6E4dwTpJ1V.png)](https://i.loli.net/2019/10/31/oUCMX6E4dwTpJ1V.png)

### 准确率和召回率
由于对每张图都检索一遍，也就是 48 次，而每张图同一类的有 6 张图， 所以需要检测到的正类为 $48\*6$ 张
同时，每次检索都寻找最相近的 6 张图，所以被检索到的图片一共有 $48\*6$ 张
这样的话， precision 和 recall 的分母都一样，而分子也一样，就变成了相同
> 不是说 precision 和 recall 是成反比的，难到不同的场景会不一样？？
```python
precision = 0.8333333333333334, recall = 0.8333333333333334
```


### 代码实现
#### 计算特征不变点
调用了OpenCV，得到所有特征不变点。此处对图片变换了尺寸，直接用原图特征点有 20 多万个，聚类需要时间过长。
opencv SIFT代码：
```python
def get_all_picture_sift_descriptor():
	PICTURE_PATH = r'picture\*'
	IMGSIZE = (300, 300)
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
```

#### 对特征不变点进行聚类
图片尺寸缩小后特征点也有 4万 多个，没有用不同 K 值进行比较，设置 K 为 1000，也没有多次随机化来防止最初随机点有问题。这样自己写的 K-means 聚类运行了 5 个多小时。
调用 scikit-learn 的 k-means 运行 20 多分钟出结果
后面计算图片特征值使用的是自己 k-means 的结果
scikit-learn的kmeans代码：
```python
K_CLUSTER = 1000
all_descriptor = get_all_picture_sift_descriptor()
kmeans = KMeans(n_clusters=K_CLUSTER, random_state=0).fit(all_descriptor)  # scikit 的K means 对 SFIT 选择的特征值聚类
centers = kmeans.cluster_centers_
```

#### 计算每张图片特征
这一步有些繁琐，重新使用了SIFT算法对图片进行计算。看着openCV SIFT算法运行很快，就没有对特征不变点备份
```python
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
```

#### 找最相近的图
```python
def find_n_nearest(feature, imgpath, picture, n):
    imgpath_distance = dict()
    n = np.minimum(feature.shape[0], n)
    for row, path in zip(feature, imgpath):
        d = L2_norm(row, picture)
        imgpath_distance[path] = d
    ans = sorted(imgpath_distance.items(), key=lambda e: e[1])
    return [x[0] for x in ans][:n]
```

#### 结果显示
将找到的最相近的 6 张图按顺序显示出来
```python
def show_picture(img_path, n_nearest_img_path):
    plt.figure()
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (100, 100))
    ax = plt.subplot2grid((3, 5), (0, 0))
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('origin img')
    for i, path in enumerate(n_nearest_img_path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (100, 100))
        ax = plt.subplot2grid((3, 5), (i//5+1, i%5))
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(str(i) +' nearest')
    plt.show()
```

#### 计算 precision 和 recall
计算 precision 和 recall 代码
```python
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
```

### 整个作业代码
```python
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
```

