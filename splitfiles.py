#!/usr/bin/env python3

indirectory = input("in directory: ")
outdirectory = input("out directory: ")
import os

segment_time = 5

count = 1

for filename in os.listdir(indirectory):
    print("hej")
    # split the file into 5 sec intervals
    command = "ffmpeg -i " + indirectory + "/" + filename + " -f segment -segment_time " + str(segment_time) + " -c copy " + outdirectory + "/out" + "{:03}".format(count) + "%03d.wav"
    os.system(command)
    print(command)
    # remove the last clip since it probably is < 5 seconds
    tempfiles = []
    for tempfile in os.listdir(outdirectory):
        if tempfile.startswith("out" + "{:03}".format(count)):
            tempfiles.append(tempfile)
    tempfiles.sort()
    removefile = tempfiles[len(tempfiles)-1]
    print(outdirectory + "/" + removefile)
    count += 1

