import cv2
from glob import glob
import numpy as np
from tqdm import tqdm
import random
import shutil

GEN_NAME_DICT = {"000": [('n', 'n', 'n'), len(glob("./data_custom/000/*.jpg"))], 
                "001": [('n', 'n', 'g'), len(glob("./data_custom/001/*.jpg"))],
                "010": [('n', 'y', 'n'), len(glob("./data_custom/010/*.jpg"))], 
                "011": [('n', 'y', 'ga'), len(glob("./data_custom/011/*.jpg"))],
                "100": [('r', 'n', 'n'), len(glob("./data_custom/100/*.jpg"))],
                "101": [('ra', 'n', 'ga'), len(glob("./data_custom/101/*.jpg"))], 
                "110": [('ra', 'y', 'n'), len(glob("./data_custom/110/*.jpg"))], 
                "111": [('r', 'y', 'g'), len(glob("./data_custom/111/*.jpg"))]
                }

def merge_image(img1, img2, img3, imgbg, mode):
    if mode == "vs":
        img_bg = imgbg.copy()
        img_bg[15:55, 10:50] = img1
        img_bg[70:110, 10:50] = img2
        img_bg[125:165, 10:50] = img3
    elif mode == "vc":
        img = np.zeros((120, 40, 3))
        img[0:40, 0:40] = img1
        img[40:80, 0:40] = img2
        img[80:120, 0:40] = img3
        img_bg = imgbg.copy()
        img_bg[30:150, 10:50] = img
    elif mode == "hs":
        img_bg = imgbg.copy()
        img_bg[10:50, 15:55] = img1
        img_bg[10:50, 70:110] = img2
        img_bg[10:50, 125:165] = img3
    elif mode == "hc":
        img = np.zeros((40, 120, 3))
        img[0:40, 0:40] = img1
        img[0:40, 40:80] = img2
        img[0:40, 80:120] = img3
        img_bg = imgbg.copy()
        img_bg[10:50, 30:150] = img
    return img_bg


def gen_data(lb_name, img_color_dict, imgbg_hor, imgbg_ver):
    k = 388
    kind1, kind2, kind3 = GEN_NAME_DICT[lb_name][0]
    img1_path_lst_choice = random.choices(img_color_dict[kind1], k=k)
    img2_path_lst_choice = random.choices(img_color_dict[kind2], k=k)
    img3_path_lst_choice = random.choices(img_color_dict[kind3], k=k)

    for step in range(k):
        img1 = cv2.imread(img1_path_lst_choice[step])
        img1 = cv2.resize(img1, (40, 40))
        img2 = cv2.imread(img2_path_lst_choice[step])
        img2 = cv2.resize(img2, (40, 40))
        img3 = cv2.imread(img3_path_lst_choice[step])
        img3 = cv2.resize(img3, (40, 40))

        for imgbg in imgbg_hor:
            img_gen_hs = merge_image(img1, img2, img3, imgbg, mode="hs")
            img_gen_hc = merge_image(img1, img2, img3, imgbg, mode="hc")
            cv2.imwrite("./data_custom/" + lb_name + "/image_" + str(GEN_NAME_DICT[lb_name][1]).zfill(6) + ".jpg", img_gen_hc)
            GEN_NAME_DICT[lb_name][1]+=1
            cv2.imwrite("./data_custom/" + lb_name + "/image_" + str(GEN_NAME_DICT[lb_name][1]).zfill(6) + ".jpg", img_gen_hs)
            GEN_NAME_DICT[lb_name][1]+=1

        for imgbg in imgbg_ver:
            img_gen_hs = merge_image(img1, img2, img3, imgbg, mode="vs")
            img_gen_hc = merge_image(img1, img2, img3, imgbg, mode="vc")
            cv2.imwrite("./data_custom/" + lb_name + "/image_" + str(GEN_NAME_DICT[lb_name][1]).zfill(6) + ".jpg", img_gen_hc)
            GEN_NAME_DICT[lb_name][1]+=1
            cv2.imwrite("./data_custom/" + lb_name + "/image_" + str(GEN_NAME_DICT[lb_name][1]).zfill(6) + ".jpg", img_gen_hs)
            GEN_NAME_DICT[lb_name][1]+=1


def main():
    root_path = "./data (copy)"
    ra_path_lst = glob(root_path+"/ra/*.jpg")
    ya_path_lst = glob(root_path+"/ya/*.jpg")
    ga_path_lst = glob(root_path+"/ga/*.jpg")
    r_path_lst = glob(root_path+"/r/*.jpg")
    y_path_lst = glob(root_path+"/y/*.jpg")
    g_path_lst = glob(root_path+"/g/*.jpg")
    n_path_lst = glob(root_path+"/n/*.jpg")

    img_color_dict = {"ra": ra_path_lst,
                    "ya": ya_path_lst,
                    "ga": ga_path_lst,
                    "r": r_path_lst,
                    "y": y_path_lst,
                    "g": g_path_lst, 
                    "n" : n_path_lst}




    imgbg_ver = []
    imgbg_hor = []
    bg_lst_path = glob("./background/*.jpg")
    for path in bg_lst_path:
        if "ver" in path:
            img = cv2.imread(path)
            imgbg_ver.append(img)
        else:
            img = cv2.imread(path)
            imgbg_hor.append(img)

    skip_lb = ["000", "001", "010", "100"]

    for lb_name in tqdm(list(GEN_NAME_DICT.keys())):
        if lb_name in skip_lb:
            continue

        gen_data(lb_name, img_color_dict, imgbg_hor, imgbg_ver)





if __name__=="__main__":
    for lb in tqdm(["001"]):
        if lb == "000":
            path_lst = glob("./data_new_2/train/n/*.jpg")
            lst = random.sample(path_lst, k=1185)
            for path in lst:
                shutil.copyfile(path, 
                                "./data_custom/" + lb + "/image_" + str(GEN_NAME_DICT[lb][1]).zfill(6) + ".jpg")
                GEN_NAME_DICT[lb][1]+=1

        elif lb == "001":
            path_lst = glob("./data_new_2/train/g/*.jpg")
            lst = random.sample(path_lst, k=2551)
            for path in lst:
                shutil.copyfile(path, 
                                "./data_custom/" + lb + "/image_" + str(GEN_NAME_DICT[lb][1]).zfill(6) + ".jpg")
                GEN_NAME_DICT[lb][1]+=1      

        elif lb == "010":
            path_lst = glob("./data_new_2/train/y/*.jpg")
            lst = random.sample(path_lst, k=6110)
            for path in lst:
                shutil.copyfile(path, 
                                "./data_custom/" + lb + "/image_" + str(GEN_NAME_DICT[lb][1]).zfill(6) + ".jpg")
                GEN_NAME_DICT[lb][1]+=1

        elif lb == "100":
            path_lst = glob("./data_new_2/train/r/*.jpg")
            lst = random.sample(path_lst, k=5365)
            for path in lst:
                shutil.copyfile(path, 
                                "./data_custom/" + lb + "/image_" + str(GEN_NAME_DICT[lb][1]).zfill(6) + ".jpg")
                GEN_NAME_DICT[lb][1]+=1         

    # main()
    for lb in list(GEN_NAME_DICT.keys()):
        print(lb, " ", len(glob("./data_custom/" + lb + "/*.jpg")))