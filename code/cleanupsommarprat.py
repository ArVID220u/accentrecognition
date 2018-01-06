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

def overlapfiles(fromdir):
    # make files overlap, which creates a lot more data artificially. should improve our accuracy a lot
    for filename in os.listdir(filedirectory):
        rfn = filedirectory + "/"  + filename + "/" + fromdir + "/sommar.mp3"
        if not os.path.exists(filedirectory + "/" + filename + "/" + "overlapfiles"):
            os.makedirs(filedirectory + "/" + filename + "/" + "overlapfiles")
        for i in range(0,6):
            nfn = filedirectory + "/" + filename + "/" + "overlapfiles" + "/" + "sommar" + str(i) + ".mp3"
            command = "ffmpeg -i " + rfn + " -ss " + str(i) + " -acodec copy " + nfn
            os.system(command)

def delfiles():
    for filename in os.listdir(filedirectory):
        os.system("rm -rf " + filedirectory + "/" + filename + "/" + "overlapfiles")

def speedupfiles(fromdir):
    # speed up some files to create a lot more artificial data. should improve accuracy perhaps
    for filename in os.listdir(filedirectory):
       dirn = filedirectory + "/" + filename + "/" + fromdir
       counter = 0
       for fn in os.lisdir(dirn):
           pass


overlapfiles("shortenedfiles")
