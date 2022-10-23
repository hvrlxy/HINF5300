import pandas as pd
import numpy as np
import os
import sys
import gzip
import shutil

# we need to initialize the path to the root of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../../"

class ProcessFile:
    def __init__(self):
        self.ROOT_DIR = ROOT_DIR
        self.data_path = ROOT_DIR + "/data/"
        pass

    def unzip_raw_files(self,folder_path: str) -> None:
        """
        Unzip all the files in the folder path. Upon running this function, 
        the original files will not be deleted, and the unzipped files will 
        be saved in the same folder.
        :param folder_path: the path to the folder from the data folder
        :return: None
        """
        full_path = ROOT_DIR + "data/" + folder_path
        for file in os.listdir(full_path):
            if file.endswith(".gz"):
                with gzip.open(full_path + file, 'rb') as f_in:
                    with open(full_path + file[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

    def get_all_subject_ids(self, folder_path: str = "raw/featuresLabel/") -> list:
        """
        Get all the subject ids from the folder path
        :param folder_path: the path to the folder from the data folder
        :return: a list of subject ids
        """
        full_path = ROOT_DIR + "data/" + folder_path
        subject_ids = []
        for file in os.listdir(full_path):
            if file.endswith(".csv"):
                subject_ids.append(file[:-4].replace(".features_labels", ""))
        return subject_ids

# test the unzip file function
if __name__ == "__main__":
    # initialize the class
    fp = ProcessFile()
    # unzip the files
    raw_folder = "raw/featuresLabel/"
    fp.unzip_raw_files(raw_folder)
    print(fp.get_all_subject_ids(raw_folder))  