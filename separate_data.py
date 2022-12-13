import cv2
from glob import glob
from tqdm import tqdm
import pandas as pd
import os

def get_image_path(path:str):
    return path.replace("labels", "images").replace(".txt", ".jpg")

def get_bboxex_and_labels(path:str, label_lst):
    bboxes = []
    labels = []
    try:
        df_txt = pd.read_csv(path, sep=" ", header=None)
    except:
        return bboxes, labels

    if not df_txt.empty:
        for idx in range(df_txt.shape[0]):
            lb_id = int(df_txt.iloc[idx][0])
            label = label_lst[lb_id]
            labels.append(label)
            box = (df_txt.iloc[idx][1], df_txt.iloc[idx][2], 
                    df_txt.iloc[idx][3], df_txt.iloc[idx][4])
            bboxes.append(box)

    return bboxes, labels


def create_data_folder(root_path:str, label_lst):
    for name in label_lst:
        path = root_path + "/" + name
        if os.path.exists(path):
            continue
        else:
            os.makedirs(path)
    print("Finished create data folder.")

def statistic_data(path:str, label_lst):
    num = len(glob(path + "/*/*.jpg"))
    print("Total images is ", num)
    for lb in label_lst:
        dt_path = path + '/' + lb + "/*.jpg"
        print("Number of images in " + lb + " class is ", len(glob(dt_path)))


def main():
    label_lst = ['g', 'ga', 'gf', 'n', 'nf', 'r', 'r-', 'ra', 'rf', 'y', 'ya', 'yf']
    create_data_folder("./data", label_lst)
    anno_path_lst = glob("./Traffic Light.v2i.yolov5pytorch/train/labels/*.txt")
    num = 0

    for anno_path in tqdm(anno_path_lst):
        img_path = get_image_path(anno_path)
        img = cv2.imread(img_path)
        bboxes, labels = get_bboxex_and_labels(anno_path, label_lst)
        for i, box in enumerate(bboxes):
            lb = labels[i]
            x = int(box[0]*img.shape[1])
            y = int(box[1]*img.shape[0])
            w = int(box[2]*img.shape[1])
            h = int(box[3]*img.shape[0])
            x1 = x - w//2
            y1 = y - h//2
            x2 = x + w//2
            y2 = y + h//2
            try: 
                sub_img = img[y1:y2, x1:x2]
                path_save = "./data/" + lb + "/image_" + str(num).zfill(6) + ".jpg"
                num+=1
                cv2.imwrite(path_save, sub_img)
            except:
                continue

    statistic_data("./data", label_lst )

if __name__ == "__main__":
    # for lb_name in ['g', 'ga', 'gf', 'n', 'nf', 'r', 'ra', 'rf', 'y', 'ya', 'yf']:
    #     path_lst = glob("./data/" + lb_name + "/*.jpg")
    #     for i, path in enumerate(path_lst):
    #         os.rename(path, "./data/" + lb_name + '/tflight_image_' + lb_name + '_' + str(i).zfill(6) + ".jpg")
    # main()
    statistic_data("./data", ['g', 'ga', 'gf', 'n', 'nf', 'r', 'ra', 'rf', 'y', 'ya', 'yf'])

