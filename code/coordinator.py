#!/usr/bin/env python3

import spectrogram as sp
import numpy as np
from pathlib import Path
import random
import network
import os
import setup
import json

"""
takes wav files and runs them through spectogram to convert them to trainging sets for nielsens code
"""

#begin with skanska


def buildpath(a, b, start, xsecfiles):
    start += xsecfiles + "/out"
    a = str(a)
    b = str(b)
    start += (3 - len(a)) * "0"
    start += a
    start += (3 - len(b)) * "0"
    start += b
    start += ".wav"
    return start

def precomppath(a, b, precomputedpath):
    start = precomputedpath + "/out"
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

def create_data_set(start_path, is_train_data, xsecfiles, time_compression=None, frequency_compression=None, cutoff=None):

    global training_data
    global test_data
    #indexes for the audio files

    correct_answer = ans(start_path)
    precomputedspectrogrampath = start_path + "precomputed_spectrograms_" + xsecfiles + "_" + str(time_compression) + "_" + str(frequency_compression) + "_" + str(cutoff)
    if time_compression == None and frequency_compression == None and cutoff == None:
        precomputedspectrogrampath = start_path + "precomputed_spectrograms_" + xsecfiles

    if (not os.path.exists(precomputedspectrogrampath) or (len(os.listdir(precomputedspectrogrampath)) == 0)):
        if not os.path.exists(precomputedspectrogrampath):
            os.makedirs(precomputedspectrogrampath)

        i = 1
        j = 0

        file_path = buildpath(i, j, start_path, xsecfiles)
        file = Path(file_path)

        global spectosize
        spectosize = -1

        while(file.is_file()):

            specto = sp.spectrogram(file_path, time_compression=time_compression, frequency_compression=frequency_compression, cutoff=cutoff)
            specto = np.reshape(specto, (specto.size, 1))
            # spectogram size needs to be the same for all spectrograms. otherwise something is terribly wrong.
            if spectosize == -1:
                spectosize = specto.size
            else:
                assert spectosize == specto.size


            #save the computed spectrograms
            np.save(precomppath(i, j, precomputedspectrogrampath), (specto, correct_answer))

            # Check if the files should be added to training_data or test_data

            if(is_train_data):
                training_data.append((specto, correct_answer))
            else:
                test_data.append((specto, correct_answer))

            file = Path(buildpath(i, j + 1, start_path, xsecfiles))

            if(file.is_file()):
                j += 1
                file_path = buildpath(i, j, start_path, xsecfiles)
            else:
                j = 0
                i += 1
                file_path = buildpath(i, j, start_path, xsecfiles)
                file = Path(file_path)

    else:
        for filename in os.listdir(precomputedspectrogrampath):

            is_train_data = (random.randint(1,10) != 1)
            #is_train_data = True

            if(is_train_data):
                training_data.append(np.load(precomputedspectrogrampath + "/" + filename))
                #test_data.append(np.load(start_path + "precomputed_spectrograms_" + xsecfiles + "/" + filename))
            else:
                test_data.append(np.load(precomputedspectrogrampath + "/" + filename))




def main():
    
    global training_data
    training_data = []

    global test_data
    test_data = []
    
    number_of_accents = 2
    number_of_test = 2

    #for filename in os.listdir(setup.DATA_PATH + "tmpvoices2"):
    #    create_data_set(setup.DATA_PATH + "tmpvoices2/" + filename + "/", False, "fivesecfiles")

    
    for filename in os.listdir(setup.DATA_PATH + "sommarprat"):
        create_data_set(setup.DATA_PATH + "sommarprat/" + filename + "/", True, "nomusic_fivesecfiles", time_compression = None, frequency_compression = None, cutoff = None)



    #for item in training_data:
    #    print(item[0].shape)

    #shuffle the training data to get random test_data
    random.shuffle(training_data)
    random.shuffle(test_data)

    #splitting training_data into test_data and training_data
#    test_data = training_data[(len(training_data) - 200):]
#    training_data = training_data[:(len(training_data) - 200)]


    #data_points are the number of input neurons
    data_points = training_data[0][0].size

    #create the network object
    net = network.Network([data_points, 100, 2])

    print(len(test_data))
    print(len(training_data))

    #trains the network with the training data
    history_data = net.SGD(training_data, epochs=200, mini_batch_size=100, eta=0.02, lmbda=0.1,
            test_data=test_data,
            monitor_training_accuracy=False,
            monitor_test_cost=False,
            monitor_training_cost=False,
            monitor_test_accuracy=True)

    
    # save the history data
    with open("lasthistory.json", "w") as f:
        json.dump(history_data, f)

    #saving the weights and biases of the trained net

    weight_file = Path("saved_weights")
    bias_file = Path("saved_biases")
    net.save("lastnetwork.json")


#    np.save(weight_file, net.weights)
#    np.save(bias_file, net.biases)


if __name__ == "__main__":
    main()
