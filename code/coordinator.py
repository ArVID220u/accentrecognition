#!/usr/bin/env python3

import spectrogram as sp
import numpy as np
from pathlib import Path
import random
import network

"""
takes wav files and runs them through spectogram to convert them to trainging sets for nielsens code
"""

#begin with skanska

training_data = []

def buildpath(a, b, start):
    start += "out"
    a = str(a)
    b = str(b)
    start += (3 - len(a)) * "0"
    start += a
    start += (3 - len(b)) * "0"
    start += b
    start += ".wav"
    return start
#This fucntion takes all files from the start_path-directory and adds them to training_data
def create_data_set(start_path, correct_answer):

    i = 1
    j = 0

    file_path = buildpath(i, j, start_path)
    file = Path(file_path)

    while(file.is_file() and i == 1):
#        print ("hej")    
        specto = sp.spectrogram(file_path)
        specto = specto.flatten()
        training_data.append((specto, correct_answer))
        
        file = Path(buildpath(i, j + 1, start_path))

        if(file.is_file()):
            j += 1
            file_path = buildpath(i, j, start_path)
        else:
            j = 0
            i += 1
            file_path = buildpath(i, j, start_path)
            file = Path(file_path)
            

create_data_set("../data/tmpvoices/skanska/fivesecfiles/", np.array([1, 0]))
create_data_set("../data/tmpvoices/stockholmska/fivesecfiles/", np.array([0, 1]))


for item in training_data:
    print(item[0].shape)

#shuffle the training data to get random test_data
random.shuffle(training_data)

#splitting training_data into test_data and training_data
test_data = training_data[(len(training_data) - 4):]
training_data = training_data[:(len(training_data) - 4)]


#print (test_data)
#print (training_data)
#data_points are the number of input neurons
data_points = training_data[0][0].size
print(data_points)
print(training_data[0][0].shape)
#create the network object
net = network.Network([data_points, 10, 10, 2])

#trains the network with the training data
net.SGD(training_data, 1, 10, 3.0, test_data=test_data)

#saving the weights and biases of the trained net
weight_file = open('saved_weigths.txt', 'w')
bias_file = open('saved_biases.txt', 'w')

for item in net.weights:
    print>>weight_file, item

for item in net.biases:
    print>>bias_file, item

