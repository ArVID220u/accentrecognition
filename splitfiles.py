#!/usr/bin/env python3

import os

def splitfiles(indirectory, outdirectory, segment_time):

    count = 1

    for filename in os.listdir(indirectory):
        if(filename == ".DS_Store"):
            continue
        print("hej")
        # split the file into 5 sec intervals
        command = "ffmpeg -i " + indirectory + "/" + filename + " -f segment -segment_time " + str(segment_time) + " -c copy " + outdirectory + "/out" + "{:03}".format(count) + "%03d.mp3"
        os.system(command)
        print(command)
        # remove the last clip since it probably is < 5 seconds
        tempfiles = []
        for tempfile in os.listdir(outdirectory):
            print(tempfile)
            if tempfile.startswith("out" + "{:03}".format(count)) and tempfile.endswith("mp3"):
                tempfiles.append(tempfile)
        print(tempfiles)
        tempfiles.sort()
        print(tempfiles)
        removefile = tempfiles[len(tempfiles)-1]
        print(removefile)
        os.remove(outdirectory + "/" + removefile)

        count += 1


    for filename in os.listdir(outdirectory):
        command = "ffmpeg -i " + outdirectory + "/" + filename + " -c:a pcm_s32le -y -ac 1 " + outdirectory + "/" + filename[:-4] + ".wav"
        os.system(command)

filedirectory = input("directory with the files: ")
rawfiledirectory = input("raw file directory: ")
xsecfiledirectory = input("xsecfiledirectory: ")
segment_time = input("segment time: ")

for filename in os.listdir(filedirectory):

    if not os.path.exists(filedirectory + "/" + filename + "/" + xsecfiledirectory):
        os.makedirs(filedirectory + "/" + filename + "/" + xsecfiledirectory)
    
    if(len(os.listdir(filedirectory + "/" + filename + "/" + xsecfiledirectory)) == 0 and len(os.listdir(filedirectory + "/" + filename + "/" + rawfiledirectory)) > 0):
        splitfiles(filedirectory + "/" + filename + "/" + rawfiledirectory, filedirectory + "/" + filename + "/" + xsecfiledirectory, segment_time)  
    print(filedirectory + "/" + filename + "/" + rawfiledirectory)


