"""
This script is a messy program to grab 160 completely random images from a folder
in my harddrive and copy them to a another folder for use for training the model
to recognize the FTC team
"""



import numpy as np
import argparse
import cv2
import os
from pathlib import Path
import imutils
from datetime import datetime
from collections import namedtuple
import pandas as pd
from random import randint
import shutil

File = namedtuple('File', 'name path size modified_date')

dest = r"C:\Users\Michael Zeng\Documents\Programming\Recognizing FTC Team\Training Images\teamFalse"
destPath = Path(dest)
allNames = []
for item in destPath.glob('**/*'):    #Loops through all files & directories in p, including nested directories
    allNames.append(item.name)

#Looping through the directories to collect all images
files = []
p = Path("D:\\Michael's iPhone6")
print(os.listdir()) #prints all directories in current directory of this file
counter = 0
for item in p.glob('**/*'):     #Loops through all files & directories in p, including nested directories
    counter += 1
    # print(counter)
    print("suffix: " + item.suffix)
    if item.suffix in ['.PNG', '.JPG', '.jpeg', '.heic']:
        name = item.name
        path = Path.resolve(item).parent
        size = item.stat().st_size
        modified = datetime.fromtimestamp(item.stat().st_mtime)
        value = randint(0, 10000)
        print("value: " + str(value))

        numFilesInFolder = len([item for item in os.listdir(r"C:\Users\Michael Zeng\Documents\Programming\Recognizing FTC Team\Training Images\teamFalse")])
        print(str(item.name in allNames))
        if value <= 1000 and item.name not in allNames:
            # files.append(File(name, path, size, modified))
            print("name: " + str(name) + "           path: " + str(path))
            os.chdir(path)
            shutil.copy(name, dest)
        if numFilesInFolder == 250:
            print("breaking")
            break
