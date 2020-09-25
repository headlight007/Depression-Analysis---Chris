# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 17:26:35 2020

@author: moore
"""
import os
from os import walk

#change path to suite your comptuter - will eventually change to extend existing path..
video_path= r"D:\OneDrive\University\2020 S2\9785 ITS Capstone Project Semester 2 2020\Projects\AVEC2013\Development\Video"
lbls_path= r"D:\OneDrive\University\2020 S2\9785 ITS Capstone Project Semester 2 2020\Projects\AVEC2013\Development\DepressionLabels"

vids = []
for (dirpath, dirnames, filenames) in walk(video_path):
    vids.extend(filenames)
    # load in image and loop over frames
        # save image
    break

labels = []
for (dirpath, dirnames, filenames) in walk(lbls_path):
    labels.extend(filenames)
    break

print(vids)
print(labels)

# LoD VIDEO GET LENGTH

# Append csv/xml/json file with the following format (i, video_file, frame_no, severity), eg: 0, video0, 0, 5; 1,  video0, 1, 5; 