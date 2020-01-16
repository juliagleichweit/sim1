import video_to_images as vti
import os.path as osp
from glob import glob

# Define the paths
base = "../../data/"
files = ["Muppets-02-01-01.avi",
         "Muppets-02-04-04.avi",
         "Muppets-03-04-03.avi"]

# do you want to get all frames as images
obtain_imgs = True
# do you want to generate your labels from your manual kermit & not_kermit categorization
generate_labels = True
processed_dir = "../../processed_img/"


def write_labels(filename:str, path:str, kermit_present: int):
    """
    Write the frames where kermit is present or not to filename

    :param filename: complete path where the file should be saved to
    :param path: has to point to kermit or not_kermit directory
    :param kermit_present: 1 is True, 0 is False
    """
    if osp.exists(path):
        file_paths = glob(osp.join(path, '*.jpg'))
        with open(filename, 'w') as f:
            for each_file in file_paths:
                frame = osp.splitext(osp.basename(each_file))[0]
                f.write(frame + " " + kermit_present.__str__() + "\n")


if __name__ == '__main__':

    if obtain_imgs:
        for file in files:
            path = base + file
            print("Processing file: ", path)
            vti.convert(processed_dir,path)
            print("File finished.")

    # you have to manually sort your frames into the
    # kermit and not_kermit directory
    # e.g. processed_img/Muppets-03-04-03/kermit has all frames where kermit is visible
    # e.g. processed_img/Muppets-03-04-03/not_kermit has all frames where kermit is not visible
    if generate_labels:
        for file in files:
            video = osp.splitext(osp.basename(file))[0]
            base = processed_dir + video
            kermit_path = base + "/kermit/"
            not_kermit_path = base + "/not_kermit/"

            # files are saved in processed_img/<filename>/<filename>_labels...
            write_labels(base+"/"+video+"_labels_kermit.txt", kermit_path, 1)
            write_labels(base+"/"+video+"_labels_not_kermit.txt", not_kermit_path, 0)

