import cv2
from glob import glob
import pandas as pd
from tqdm import tqdm
import random
import shutil

def get_name_image_and_label(path):
    return path.split('/')[-1].split('.')[0], path.split('/')[-2]

def remove_list(root_lst:list, sub_lst:list):
    for item in sub_lst:
        root_lst.remove(item)
    return root_lst

def data_split(lst:list, raito):
    train_lst = random.sample(lst, k=int(raito[0]*len(lst)))
    lst = remove_list(lst, train_lst)
    val_lst = random.sample(lst, k=int(raito[1]/(raito[1]+raito[2])*len(lst)))
    test_lst = remove_list(lst, val_lst)
    return train_lst, val_lst, test_lst

def main():
    lb_lst = ["000", "001", "010", "011", "100", "101", "110", "111"]
    df_train = pd.DataFrame(columns=["Filenames", 'r', 'y', 'g'])
    df_val = pd.DataFrame(columns=["Filenames", 'r', 'y', 'g'])
    df_test = pd.DataFrame(columns=["Filenames", 'r', 'y', 'g'])

    for lb_name in tqdm(lb_lst):
        x1, x2, x3 = list(lb_name)
        lb_lst = glob("./data_custom/" + lb_name + "/*.jpg")
        train_lst, val_lst, test_lst = data_split(lb_lst, (0.7, 0.2, 0.1))
        for data_path in train_lst:
            name = data_path.split('/')[-1]
            shutil.copyfile(data_path, "./data_total/train/images/"+lb_name + "_" + name)
            df = pd.DataFrame([[lb_name + "_" +name, x1, x2, x3]], columns=["Filenames", 'r', 'y', 'g'])
            df_train = pd.concat([df_train, df], ignore_index=True)

        for data_path in val_lst:
            name = data_path.split('/')[-1]
            shutil.copyfile(data_path, "./data_total/val/images/"+lb_name + "_" +name)
            df = pd.DataFrame([[lb_name + "_" +name, x1, x2, x3]], columns=["Filenames", 'r', 'y', 'g'])
            df_val = pd.concat([df_val, df], ignore_index=True)

        for data_path in test_lst:
            name = data_path.split('/')[-1]
            shutil.copyfile(data_path, "./data_total/test/images/"+lb_name + "_" +name)
            df = pd.DataFrame([[lb_name + "_" +name, x1, x2, x3]], columns=["Filenames", 'r', 'y', 'g'])
            df_test = pd.concat([df_test, df], ignore_index=True)

    df_train.to_csv("./data_total/train/train.csv", index=False)
    df_val.to_csv("./data_total/val/val.csv", index=False)
    df_test.to_csv("./data_total/test/test.csv", index=False)

    print("Finished.")



if __name__ == "__main__":
    # main()

    for phase in ["train", "val", "test"]:
        print(phase, " ", len(glob("./data_total/" + phase + "/images/*.jpg")))

