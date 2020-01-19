import shutil, os
import os.path as osp
from gen_img_labels import GenerateFramesLabels

"""
Purpose is 

    - to extract the frames from the data path
    - generate labels AFTER manual sorting into kermit/not_kermit
    - move the labeled .jpg files to the corresponding train and test directories

This is done to use the keras.ImageDataGenerator() to automatically
load the images and generate the labels according to the directory structure
e.g. data structure, where you copy your train+val frames into the train dir
and the rest in the test dir

The ImageDataGenerator loads the images and automatically labels them by the directory
name. This means the images in the kermit dir are labeled as kermit and 
the ones in the not_kermit folder are labeled as not_kermit

train
- kermit
- not_kermit

test
- kermit
- non_kermit
"""

# Define the paths
base = "../../data/"
processed_dir = "../../processed_img/"

dirs_train = ["Muppets-02-01-01", "Muppets-03-04-03"]
dirs_test = ["Muppets-02-04-04"]

train_dir = "../../data/train"
test_dir = "../../data/test"

pattern = "*.jpg"
"""


	SET THESE VARIABLES ACCORRDING TO YOUR PREFERENCES


"""
# do you want to get all frames as images
obtainImgs= True

# do you want to generate your labels from your manual kermit & not_kermit categorization
# files are saved in the same folder as this script
genLab = False

# do you want to move the files to train/test dirs
moveFiles = True

def check_dir_struc(dir:str):
    # Create dir for video file if not existent
    # this is where we save our images
    if not osp.exists(dir):
        os.mkdir(dir)
        os.mkdir(dir + "/kermit/")
        os.mkdir(dir + "/not_kermit/")


def move(frame_file:str, class_dir:str, subdir:str, move_to_dir:str):
    """
    Move the files from the processed_img to the train or test dir

    :param frame_file: path to _labels_kermit.txt or ...labels_not_kermit.txt
    :param class_dir: path to <muppet_dir>/kermit or <muppet_dir>/not_kermit
    :param subdir: "kermit" or "not_kermit"
    :param move_to_dir: train_dir or test_dir
    """
    with open(frame_file) as my_file:
        for frame in my_file:
            file_name = frame.strip()
            src = osp.join(class_dir, file_name + ".jpg")
            dst = osp.join(move_to_dir, subdir, file_name + ".jpg")
            if osp.exists(src):
                shutil.move(src, dst)

def exec_for_dir(dirs:str, tt_dir:str):
    """
    Move the kermit/non_kermit imgs for each train dir or test dir

    :param dirs:  array specifying the <muppet-file> dirs for training or test
    :param tt_dir:  train_dir or test_dir
    :return:
    """
    for muppet_dir in dirs:
        # kermit_file = osp.join(processed_dir, muppet_dir)
        # path_kermit = osp.join(kermit_file, "kermit")
        path_kermit = osp.join(processed_dir, muppet_dir)
        # kermit_file = osp.join(kermit_file, muppet_dir+"_labels_kermit.txt")
        kermit_file = muppet_dir+"_labels_kermit.txt"
        print("reading file: ", kermit_file)

        #not_kermit_file = osp.join(processed_dir, muppet_dir)
        # path_not_kermit = osp.join(not_kermit_file, "not_kermit")
        path_not_kermit = osp.join(processed_dir, muppet_dir)
        # not_kermit_file = osp.join(not_kermit_file, muppet_dir+"_labels_not_kermit.txt")
        not_kermit_file = muppet_dir+"_labels_not_kermit.txt"
        print("reading file: ", not_kermit_file)

        move(kermit_file,path_kermit,"kermit", tt_dir)
        move(not_kermit_file,path_not_kermit,"not_kermit", tt_dir)


if __name__ == '__main__':

    frame_gen = GenerateFramesLabels(processed_dir=processed_dir, data_dir=base)
    files = dirs_train + dirs_test
    frame_gen.exec(obtain_imgs=obtainImgs, generate_labels=genLab, files=files)

    if moveFiles:
        print("moving files ...")
        # create directory structure if missing
        check_dir_struc(train_dir)
        check_dir_struc(test_dir)

        exec_for_dir(dirs_train, train_dir)
        exec_for_dir(dirs_test, test_dir)
        print("done")
