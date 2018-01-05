#!/usr/bin/env python3

import spectrogram as sp
import numpy as np
from pathlib import Path
import random
import network
import os

"""
takes wav files and runs them through spectogram to convert them to trainging sets for nielsens code
"""

#begin with skanska


def buildpath(a, b, start):
    start += "fivesecfiles/out"
    a = str(a)
    b = str(b)
    start += (3 - len(a)) * "0"
    start += a
    start += (3 - len(b)) * "0"
    start += b
    start += ".wav"
    return start

def precomppath(a, b, start):
    start += "precomputed_spectrograms/out"
    a = str(a)
    b = str(b)
    start += (3 - len(a)) * "0"
    start += a
    start += (3 - len(b)) * "0"
    start += b
    return start

def ans(path):
    
    with open(path + "accent.txt") as a:
        classification = a.read()

    #remove the newline at the end of a
    classification = classification[:-1]

    print(path + "   " + classification)
    if classification == "skanska":
        return np.array(np.reshape([0, 1], (2, 1)))
    elif classification == "stockholmska":
        return np.array(np.reshape([1, 0], (2, 1)))
    else:
        import sys
        sys.exit("ERROR: Classification needs to be either skanska or stockholmska. Line 36.")
        
"""
This fucntion takes all files from the start_path-directory and adds them to training_data.
start_path is the directory where the files are situated and is_train_data is a boolean that decides whether 
the data is going to be used for testing och training.
"""

def create_data_set(start_path, is_train_data):

    global training_data
    global test_data
    #indexes for the audio files

    correct_answer = ans(start_path)

    if (not os.path.exists(start_path + "precomputed_spectrograms")) or (len(os.listdir(start_path + "precomputed_spectrograms")) == 0):
        if not os.path.exists(start_path + "precomputed_spectrograms"):
            os.makedirs(start_path + "precomputed_spectrograms")

        i = 1
        j = 0

        file_path = buildpath(i, j, start_path)
        file = Path(file_path)

        while(file.is_file()):
    #        print ("hej")    
            specto = sp.spectrogram(file_path)
            specto = np.reshape(specto, (12840, 1))

            #save the computed spectrograms
            np.save(precomppath(i, j, start_path), (specto, correct_answer))

            # Check if the files should be added to training_data or test_data

            if(is_train_data):
                training_data.append((specto, correct_answer))
            else:
                test_data.append((specto, correct_answer))

            file = Path(buildpath(i, j + 1, start_path))

            if(file.is_file()):
                j += 1
                file_path = buildpath(i, j, start_path)
            else:
                j = 0
                i += 1
                file_path = buildpath(i, j, start_path)
                file = Path(file_path)

    else:
        for filename in os.listdir(start_path + "precomputed_spectrograms"):

            if(is_train_data):
                training_data.append(np.load(start_path + "precomputed_spectrograms/" + filename))
            else:
                test_data.append(np.load(start_path + "precomputed_spectrograms/" + filename))


debug = True


def main():
    
    global training_data
    training_data = []

    global test_data
    test_data = []
    
    number_of_accents = 2
    number_of_test = 2

    
    for filename in os.listdir("../data/sommarprat_test_data"):
        create_data_set("../data/sommarprat_test_data/" + filename + "/", False)

    counter = 0
    for filename in os.listdir("../data/sommarprat"):
        create_data_set("../data/sommarprat/" + filename + "/", True)
        counter += 1
        if debug and counter > 1:
            # debug means we want fast results
            break



    #for item in training_data:
    #    print(item[0].shape)

    #shuffle the training data to get random test_data
    random.shuffle(training_data)
    random.shuffle(test_data)

    #splitting training_data into test_data and training_data
   # test_data = training_data[(len(training_data) - 100):]
    #training_data = training_data[:(len(training_data) - 100)]


    #data_points are the number of input neurons
    data_points = training_data[0][0].size

    #create the network object
    net = network.Network([data_points, 15, 15, 15, 2])

    #trains the network with the training data
    net.SGD(training_data, 7, 200, 20.0, test_data=test_data)
    net.SGD(training_data, 7, 200, 30.0, test_data=test_data)
    net.SGD(training_data, 7, 200, 40.0, test_data=test_data)
    net.SGD(training_data, 7, 200, 10.0, test_data=test_data)
    net.SGD(training_data, 7, 200, 50.0, test_data=test_data)


    #saving the weights and biases of the trained net

    weight_file = Path("saved_weights")
    bias_file = Path("saved_biases")

    np.save(weight_file, net.weights)
    np.save(bias_file, net.biases)


if __name__ == "__main__":
    main()
