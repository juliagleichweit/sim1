import os
import os.path as osp
import cv2 as cv
import tqdm


def convert(processed_dir: str, path: str, frame_rate: float = 0.5):
    """
    Convert a video file to series of images
    Images are saved to processed_img/filename

    :param processed_dir : dir where the images should be stored
    :param path: to the image file
    :param frame_rate: how many frames per second (default is 2 FPS)
    :return:
    """

    video_file = path
    out_dir = processed_dir + osp.splitext(osp.basename(video_file))[0]

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
    # we want to save the exact time for each frame
    timestamps = []

    sec = 0
    total_count = (1/frame_rate)*((60*25)+50)
    pbar = tqdm.tqdm(total=total_count, leave=False)

    count = 0
    while (cap.isOpened()):
        # advance to next frame position
        cap.set(cv.CAP_PROP_POS_MSEC,sec*1000)
        frame_exists, curr_frame = cap.read()

        if frame_exists:
            timestamp = cap.get(cv.CAP_PROP_POS_MSEC)

            # output is : video_file/frameNr_TimestampMS.jpg
            #cv.imwrite(osp.join(out_dir, '{}_{:0<8.2f}.jpg'.format(count, timestamp)), curr_frame)
            cv.imwrite(osp.join(out_dir, '{}.jpg'.format(count)), curr_frame)
        else:
            break

        count = count + 1
        sec = sec + frame_rate
        sec = round(sec, 2)
        pbar.update(1)

    pbar.close()
    # release resources
    cap.release()

