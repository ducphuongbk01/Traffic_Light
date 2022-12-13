from glob import glob
import cv2
from tqdm import tqdm
import numpy as np

def change_color(img_path, color, dst):
    img = cv2.imread(img_path)
    lst = []
    name = img_path.split('/')[-1].split('.')[0]
    if color=='r':
        lst = range(0, 10, 1)
    elif color=='y':
        lst = range(20,30,1)
    elif color=='g':
        lst = range(45, 90, 5)
    for i, hue_val in enumerate(lst):
        path_save = "./data_generate/"+dst+'/'+name+'_'+str(i)+".jpg"
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img_hsv)

        diff_color = hue_val - h

        hnew = np.mod(h + diff_color, 180).astype(np.uint8)

        hsv_new = cv2.merge([hnew,s,v])
        bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
        cv2.imwrite(path_save, bgr_new)

def main():
    lst_lb = ['g', 'ga', 'r', 'ra', 'y', 'ya']
    root_data_path = "./data"
    for lb in lst_lb:
        img_path_lst = glob(root_data_path+'/'+lb+"/*.jpg")
        for img_path in tqdm(img_path_lst):
            if lb in ['g', 'ga', 'gf']:
                change_color(img_path, 'r', 'r' if len(lb)<2 else 'r'+lb[1])
                change_color(img_path, 'y', 'y' if len(lb)<2 else 'y'+lb[1])
            elif lb in ['y', 'ya', 'yf']:
                change_color(img_path, 'r', 'r' if len(lb)<2 else 'r'+lb[1])
                change_color(img_path, 'g', 'g' if len(lb)<2 else 'g'+lb[1])
            elif lb in ['r', 'ra', 'rf']:
                change_color(img_path, 'y', 'y' if len(lb)<2 else 'y'+lb[1])
                change_color(img_path, 'g', 'g' if len(lb)<2 else 'g'+lb[1])
        


if __name__ == "__main__":
    main()