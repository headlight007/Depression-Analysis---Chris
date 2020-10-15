# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:26:35 2020

@author: moore
"""
import os
from os import walk
import pandas as pd
import glob

#change path to suite your comptuter - will eventually change to extend existing path..
video_path= r"D:\OneDrive\University\2020 S2\9785 ITS Capstone Project Semester 2 2020\AVEC2013\Development\Video"
lbls_path= r"D:\OneDrive\University\2020 S2\9785 ITS Capstone Project Semester 2 2020\AVEC2013\Development\DepressionLabels"
frame_path= r"C:\Users\moore\Documents\GitHub\Depression-Analysis-Project\Chris-PythonTrials\frames"

# vids = []
# for (dirpath, dirnames, filenames) in walk(video_path):
#     vids.extend(filenames)
#     # load in image and loop over frames
#         # save image
#     break

# labels = []
# for (dirpath, dirnames, filenames) in walk(lbls_path):
#     labels.extend(filenames)
#     break


depressionIndex = {}
for (dirpath, dirnames, filenames) in walk(lbls_path):
    for file in filenames:
        with open(lbls_path+"\\" + (file)) as csv:
            val=int(csv.read())
           # depressionIndex[file[:5]]=val
            depressionIndex[file.split('_D')[0]]=val
 
vids = []
for (dirpath, dirnames, filenames) in walk(video_path):
    for file in filenames:
        vids.append(dirpath + "\\" + file)
        print(vids[-1])
        print(depressionIndex[file.split('_c')[0]])
    # load in image and loop over frames
        # save image

frames = []
for (dirpath, dirnames, filenames) in walk(frame_path):
    for file in filenames:
        frames.append(file)
        print(frames[-1])
    # load in image and loop over frames
        # save image
            
#print(video_path)
#print(vids)
print()
#print(lbls_path)
#print(depressionIndex)

# LoD VIDEO GET LENGTH

# Append csv/xml/json file with the following format (i, video_file, frame_no, severity), eg: 0, video0, 0, 5; 1,  video0, 1, 5; 