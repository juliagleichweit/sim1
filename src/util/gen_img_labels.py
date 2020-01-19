import video_to_images as vti
import os.path as osp
from glob import glob


class GenerateFramesLabels:

    def __init__(self, processed_dir: str, data_dir: str):
        """

        :param processed_dir: path to dir where frames should be stored
        :param data_dir: path where data files are
        """
        self._processed_dir = processed_dir
        self._base = data_dir

    def _write_labels(filename:str, path:str, kermit_present: int):
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
                    #f.write(frame + " " + str(kermit_present) + "\n")
                    f.write(frame + "\n")

    def exec(self, obtain_imgs: bool,  generate_labels: bool, files: [str]):
        """
        Generate the frames from the video files in files arr and/or generate labels
        according to prior manual sorting in kermit/not_kermit

        :param obtain_imgs: bool, extract frames (1 FPS)
        :param generate_labels: bool,  you have to manually sort your frames into the
                                kermit and not_kermit directory
                                e.g. processed_img/Muppets-03-04-03/kermit has all frames where kermit is visible
                                e.g. processed_img/Muppets-03-04-03/not_kermit has all frames where kermit is not visible
        :param files: str array containing Muppets-XX-XX-XX
        :return:
        """
        if obtain_imgs:
            print("obtaining frames...")
            for file in files:
                path = self._base + file + ".avi"
                print("Processing file: ", path)
                vti.convert(self._processed_dir,path)
                print("File finished.")

        # you have to manually sort your frames into the
        # kermit and not_kermit directory
        # e.g. processed_img/Muppets-03-04-03/kermit has all frames where kermit is visible
        # e.g. processed_img/Muppets-03-04-03/not_kermit has all frames where kermit is not visible
        if generate_labels:
            print("generating labels...")
            for file in files:
                video = osp.splitext(osp.basename(file))[0]
                base = self._processed_dir + video
                kermit_path = osp.join(base, "kermit")
                not_kermit_path = osp.join(base, "not_kermit")

                # files are saved in processed_img/<filename>/<filename>_labels...
                self._write_labels(video+"_labels_kermit.txt", kermit_path, 1)
                self._write_labels(video+"_labels_not_kermit.txt", not_kermit_path, 0)