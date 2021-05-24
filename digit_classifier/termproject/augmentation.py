import torch
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
import glob

# visualization
import matplotlib.pyplot as plt

def save_as_csv(data, label, name):
    df_dataset = pd.DataFrame(np.hstack((label.reshape(-1,1), data.reshape(-1, 784*3))))
    df_dataset.to_csv("./dataset/{0}.csv".format(name), index=False)

def show_img(data, label):
    r = np.random.randint(0, len(data))
    print(label[r])
    plt.imshow(data[r].reshape(28,28,-1))

def randomly_sampling(df_data, num=13000): # data type is DataFrame
    df_data = shuffle(df_data)
    shuffled_label = np.array(df_data["0"][:num])
    shuffled_data = np.array(df_data.drop("0", axis=1)[:num], dtype=np.uint8)
    return shuffled_label, shuffled_data

def rotation(data, num): # (-1, 784)
    """rotate several directions"""
    rotated_data = []
    h, w = 28, 28
    cX, cY = h//2, w//2
    degrees = [-10, -11, -12, -13, -14, -15, 10, 11, 12, 13, 14, 15]
    for n in range(num):
        degree = np.random.choice(degrees)
        rotation_matrix = cv2.getRotationMatrix2D((cX, cY), degree, 1.0)
        for i in range(len(data)):
            rotated_data.append(cv2.warpAffine(data[i].reshape(28,28), rotation_matrix, (w, h)))
        
    return np.array(rotated_data) # (-1, 784)

def colouring(data, num): # (-1, 784)
    """natual color"""
    colored_data = []
    for n in range(num):
        for i in range(len(data)):
            r_rand = np.random.randint(0, 256)
            g_rand = np.random.randint(0, 256)
            b_rand = np.random.randint(0, 256)

            r_bg_rand = np.random.randint(0, 256)
            g_bg_rand = np.random.randint(0, 256)
            b_bg_rand = np.random.randint(0, 256)

            while True:
                if (r_bg_rand != r_rand) and (g_bg_rand != g_rand) and (b_bg_rand != b_rand):
                    break
                r_bg_rand = np.random.randint(0, 256)
                g_bg_rand = np.random.randint(0, 256)
                b_bg_rand = np.random.randint(0, 256)

            r = np.where(data[i]>0, data[i]+r_rand, r_bg_rand)
            g = np.where(data[i]>0, data[i]+g_rand, g_bg_rand)
            b = np.where(data[i]>0, data[i]+b_rand, b_bg_rand)

            colored_data.append(np.dstack((r,g,b)))
    
    return np.array(colored_data).reshape(-1, 784*3)

def coloring(data, num): # (-1, 784)
    """solid color"""
    colored_data = []
    for n in range(num):
        for i in range(len(data)):
            r_rand = np.random.randint(0, 256)
            g_rand = np.random.randint(0, 256)
            b_rand = np.random.randint(0, 256)

            r_bg_rand = np.random.randint(0, 256)
            g_bg_rand = np.random.randint(0, 256)
            b_bg_rand = np.random.randint(0, 256)

            while True:
                if (r_bg_rand != r_rand) and (g_bg_rand != g_rand) and (b_bg_rand != b_rand):
                    break
                r_bg_rand = np.random.randint(0, 256)
                g_bg_rand = np.random.randint(0, 256)
                b_bg_rand = np.random.randint(0, 256)

            r = np.where(data[i]>0, r_rand, r_bg_rand)
            g = np.where(data[i]>0, g_rand, g_bg_rand)
            b = np.where(data[i]>0, b_rand, b_bg_rand)

            colored_data.append(np.dstack((r,g,b)))
    
    return np.array(colored_data).reshape(-1, 784*3)

def shift(data, num): # (-1, 784*3)
    """shift several ways"""
    shifted_data = []
    coef = [(-np.random.randint(3, 6), -np.random.randint(3, 6)),
            (np.random.randint(3, 6), np.random.randint(3, 6)),
            (np.random.randint(3, 6), -np.random.randint(3, 6)),
            (-np.random.randint(3, 6), np.random.randint(3, 6)),
            (0, np.random.randint(3, 6)),
            (np.random.randint(3, 6), 0),
            (0, -np.random.randint(3, 6)),
            (-np.random.randint(3, 6), 0)]
    for n in range(num):
        for i in range(len(data)):
            M = np.float32([[1,0,coef[n][0]],[0,1,coef[n][1]]]) # x axis, y axis
            dst = cv2.warpAffine(data[i].reshape(28,28,-1), M, (28, 28))
            shifted_data.append(dst)
    
    return np.array(shifted_data) # (-1, 784*3)

def expand(data): # (-1, 784*3)
    """scale image"""
    expanded_data = []
    for i in range(len(data)):
        expanded = cv2.resize(data[i].reshape(28,28,-1), dsize=(0,0), fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
        cropped = expanded[4:4+28, 4:4+28]
        expanded_data.append(cropped)
        
    return np.array(expanded_data) # (-1, 784*3)

def shrink(data): # (-1, 784*3)
    """shrink image"""
    expanded_data = []
    for i in range(len(data)):
        tmp = data[i].reshape(28,28,-1)
        expanded = cv2.resize(tmp, dsize=(0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        value = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
        pad = cv2.copyMakeBorder(expanded, 7, 7, 7, 7, cv2.BORDER_CONSTANT, value=value)
        expanded_data.append(pad)
        
    return np.array(expanded_data) # (-1, 784*3)

def img_invert(data): # (-1, 784)
    """invert image"""
    inverted = []
    for i in range(len(data)):
        inverted.append(cv2.bitwise_not(data[i].reshape(28,28,-1)))
        
    return np.array(inverted).reshape(-1, 784)

def gray2RGB(data):
    """make 28x28x1 MNIST image to 28x28x3"""
    rgb_data = []
    for i in range(len(data)):
        rgb_data.append(cv2.cvtColor(data[i], cv2.COLOR_GRAY2RGB))
    return np.array(rgb_data).reshape(-1,784*3)