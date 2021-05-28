import numpy as np
import argparse
import cv2
import os
from pathlib import Path
import imutils
from datetime import datetime
from collections import namedtuple
import pandas as pd

#Looping through the directories to collect all images
files = []
p = Path("D:\\Michael's iPhone6")
print(os.listdir()) #prints all directories in current directory of this file
for item in p.glob('**/*'):     #Loops through all files & directories in p, including nested directories
    if item.suffix in ['.png', '.jpg', '.jpeg', '.heic']:
        name = item.name
        path = Path.resolve(item).parent
        size = item.stat().st_size
        modified = datetime.fromtimestamp(item.stat().st_mtime)
        files.append(File(name, path, size, modified))
print(files)
