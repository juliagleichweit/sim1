import os
import os.path as osp
import cv2 as cv
import tqdm
import math

def convert(processed_dir: str, video_file: str):
    """
    Convert a video file to series of images
    Images are saved to processed_img/filename

    :param processed_dir : dir where the images should be stored
    :param video_file: path to the image file
    :param frame_rate: how many frames per second (default is 2 FPS)
    :return:
    """

    video_name = osp.splitext(osp.basename(video_file))[0]
    out_dir = processed_dir + video_name

    # create img dir
    if not osp.exists(processed_dir):
        os.mkdir(processed_dir)

    # Create dir for video file if not existent
    # this is where we save our images
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    if osp.exists(out_dir):
        os.mkdir(out_dir + "/kermit/")
        os.mkdir(out_dir + "/not_kermit/")

    # open video file for processing
    cap = cv.VideoCapture(video_file)
    frame_rate = cap.get(5)  # frame rate

    sec = 0
    total_count = (60*25)+50 # just an approximation
    pbar = tqdm.tqdm(total=total_count, leave=False)

    count = 0
    while (cap.isOpened()):
        frame_id = cap.get(1)  # current frame number
        frame_exists, curr_frame = cap.read()

        if not frame_exists:
            break
        else:
            if (frame_id % math.floor(frame_rate) == 0):
                # output is : video_file/<video_file>_frameNr.jpg
                cv.imwrite(osp.join(out_dir, '{}_{}.jpg'.format(video_name,count)), curr_frame)
                count = count + 1
                pbar.update(1)

    pbar.close()
    # release resources
    cap.release()

