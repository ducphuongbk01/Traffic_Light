import numpy as np
import cv2
import tensorflow as tf
import albumentations as A
import random
from glob import glob
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import shutil

# Data Generate
data_transform = A.Compose([
                            A.VerticalFlip(p=0.7),
                            A.HorizontalFlip(p=0.7),
                            A.RandomSnow(p=0.7),
                            A.RandomRain(p=0.7),
                            A.augmentations.geometric.rotate.Rotate(limit=[-30,30],p=0.7),
                            A.augmentations.transforms.MotionBlur(blur_limit=7, p=0.7),
                            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7)
                          ])

# Ham can kiem tra
def get_file_name(path):
    return path.split('\\')[-1].split('.')[0]


def generateData(data_lst:list, lb_lst:list, ratio=2):
    data_lst_tmp = data_lst.copy()
    lb_lst_tmp = lb_lst.copy()
    for i, img in enumerate(data_lst):
        for j in range(ratio-1):
            img2 = data_transform(image=img)["image"]
            data_lst_tmp.append(img2)
            lb_lst_tmp.append(lb_lst[i])
    return data_lst_tmp, lb_lst_tmp


def balanceData(data_path:str):

    data_path_lst = {'r': glob(data_path + "/r/*jpg"),
                     'y': glob(data_path + "/y/*jpg"),
                     'g': glob(data_path + "/g/*jpg"),
                     'n': glob(data_path + "/n/*jpg")
                    }

    num_r = len(data_path_lst['r'])
    num_y = len(data_path_lst['y'])
    num_g = len(data_path_lst['g'])
    num_n = len(data_path_lst['n'])

    num_max = max([num_r, num_y, num_g, num_n])

    for num, lb in zip((num_r, num_y, num_g, num_n), ('r', 'y', 'g', 'n')):
        diff_num = num_max - num
        lst_path = random.choices(data_path_lst[lb], k=diff_num)

        for i, path in enumerate(lst_path):

            try:
                name = get_file_name(path)

                new_save_path = data_path + "/" + lb + "/" + "data_generate_" + str(i) + "_" + name + ".jpg" 

                img = cv2.imread(path)
                img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                img2 = data_transform(image=img1)["image"]
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

                cv2.imwrite(new_save_path, img2)
            except:
                print(path)

    num_r_after = len(glob(data_path + "/r/*jpg"))
    num_y_after = len(glob(data_path + "/y/*jpg"))
    num_g_after = len(glob(data_path + "/g/*jpg"))
    num_n_after = len(glob(data_path + "/n/*jpg"))

    print("Before balance...")
    print("Red class: " + str(num_r))
    print("Yellow class: " + str(num_y))
    print("Green class: " + str(num_g))
    print("None class: " + str(num_n))
    print("After balance...")
    print("Red class: " + str(num_r_after))
    print("Yellow class: " + str(num_y_after))
    print("Green class: " + str(num_g_after))
    print("None class: " + str(num_n_after))

    print("------------Finished--------------")


def data_filter(data_path:str, threshold:int):
    data_path_lst = glob(data_path + "/*/*.jpg")

    for path in tqdm(data_path_lst):
        img = cv2.imread(path)
        if min(img.shape[0:2]) < threshold:
            os.remove(path)

def image_size_statistic(data_path:str):
    data_path_lst = glob(data_path + "/*/*.jpg")

    arr = []

    for path in tqdm(data_path_lst):
        img = cv2.imread(path)

        min_d = min(img.shape[0:2])
        arr.append(min_d)

    # Creating histogram
    fig, ax = plt.subplots(figsize =(10, 7))
    ax.hist(arr, bins = range(0, 10, 1))
    
    # Show plot
    plt.show()


def separate_data(raito:float, data_path:str):
    data_path_lst = {'r': glob(data_path + "/r/*jpg"),
                     'y': glob(data_path + "/y/*jpg"),
                     'g': glob(data_path + "/g/*jpg"),
                     'n': glob(data_path + "/n/*jpg")
                    }

    num = int(raito*len(data_path_lst['r']))

    for lb in ('r', 'y', 'g', 'n'):
        data_lst = data_path_lst[lb]

        lst = random.sample(data_lst, k=num)
        for path in tqdm(lst):
            dest = path.replace('total', 'train')
            shutil.move(path, dest)
    
    os.rename(data_path, data_path.replace("total", "validation"))
    print("Finished...")

def check_data(data_path:str):
    for phase in ("train", "validation"):
        print(phase + ":")
        print("Red class: " + str(len(glob(data_path + "/" + phase + "/r/*jpg"))))
        print("Yellow class: " + str(len(glob(data_path + "/" + phase + "/y/*jpg"))))
        print("Green class: " + str(len(glob(data_path + "/" + phase + "/g/*jpg"))))
        print("None class: " + str(len(glob(data_path + "/" + phase + "/n/*jpg"))))

def generate_class(data_path, class_name, raito):
    data_class_lst = glob(data_path + '/' + class_name + "/*.jpg")
    path_save = data_path + '/' + class_name

    data_choice_lst = random.sample(data_class_lst, k=int(raito*len(data_class_lst)))

    for i, path in enumerate(data_choice_lst):
        try:
            name = get_file_name(path)

            img = cv2.imread(path)
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img2 = data_transform(image=img1)["image"]
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            path_save_img = path_save + "/gen_class_" + name + '_' + str(i) + ".jpg"
            cv2.imwrite(path_save_img, img2)
        except:
            print(path)
    

def main():
    data_path = "/home/phuongdoan/Phuong/Traffic_Project"

    check_data(data_path)



if __name__ == "__main__":
    # generate_class("./data_new/train", 'y', 0.2)
    check_data("./data_new")
