#!/usr/bin/env python3
import os
import setup

# The purpose of this file is to clean up sommarprat data

filedirectory = setup.DATA_PATH + "sommarprat"

def rename():
    # rename so all files are formatted "sommarprat.mp3"
    for filename in os.listdir(filedirectory):
        rfn = filedirectory + "/" + filename + "/" + "superrawfiles"
        for ffname in os.listdir(rfn):
            os.system("mv " + rfn + "/" + ffname + " " + rfn + "/" + "sommar.mp3")



def shortenbeginning():
    # remove 25 seconds in beginning
    for filename in os.listdir(filedirectory):
        rfn = filedirectory + "/" + filename + "/superrawfiles/sommar.mp3"
        if not os.path.exists(filedirectory + "/" + filename + "/" + "shortenedfiles"):
            os.makedirs(filedirectory + "/" + filename + "/" + "shortenedfiles")
        nfn = filedirectory + "/" + filename + "/shortenedfiles/sommar.mp3"
        start_time = 45
        command = "ffmpeg -y -i " + rfn + " -ss " + str(start_time) + " -acodec copy " + nfn
        os.system(command)

shortenbeginning()
