#!/usr/bin/env python3
#coding: utf-8
""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from math import *

#plt.rc('text', usetex=True)
plt.rc('font', family='serif')

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    #cols = 250
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(int(cols), int(frameSize)), strides=(int(samples.strides[0]*hopSize), int(samples.strides[0]))).copy()
    #print("frames shape: " + str(np.shape(frames)))
    frames *= win
    
    return np.fft.rfft(frames)    


def spectrogram(audiopath, time_compression=10, frequency_compression=10, cutoff=300):
    if time_compression == None:
        time_compression = 10
    if frequency_compression == None:
        frequency_compression = 10
    if cutoff == None:
        cutoff = 300
    binsize = 2**11
    samplerate, samples = wav.read(audiopath)
    #print(len(samples))
    samples = samples[:219136]
    samples = samples / (2.**15)
    #print(samplerate)
    #print(samples.dtype)
    #print("shape of wav samples: " + str(np.shape(samples)))
    s = stft(samples, binsize)

    #print("timebins: " + str(np.shape(s)[0]))
    #print("freqbins: " + str(np.shape(s)[1]))

    n = binsize
    nUniquePts = int(ceil((n+1)/2.0))
    s = abs(s)

    s = s / float(n)
    s = s**2

    #s[:,arange(1,len(s)-1)] = s[:,arange(1, len(s) - 1))
    s[:,1:len(s)-1] = s[:,1:len(s)-1]*2
#    print("shape of s: " + str(np.shape(s)));

    ng = []
    maxx = 0

    # compress times
    cur_time = []
    count = 0

    for r in s:
        
        i = 0
        rg = []
        while i < cutoff:

            # compress data points into one data point by averaging them
            # this compresses frequencies
            avg = 0
            ni = i
            while ni < len(r) and ni - i < frequency_compression:
                avg += r[ni]
                ni += 1
            avg /= (ni-i)
            i = ni
            rg.append(avg)

        if count >= time_compression:
            # compress the cur_time
            ddg = cur_time[0]
            for dddd in cur_time[1:]:
                assert len(dddd) == len(ddg)
                for ii in range(len(dddd)):
                    ddg[ii] += dddd[ii]
            for ii in range(len(ddg)):
                ddg[ii] = ddg[ii] / len(cur_time)
            ng.append(ddg)
            cur_time = []
            cur_time.append(rg)
            count = 1
        else:
            cur_time.append(rg)
            count += 1

    # add last cur_time
    ddg = cur_time[0]
    for dddd in cur_time[1:]:
        assert len(dddd) == len(ddg)
        for ii in range(len(dddd)):
            ddg[ii] += dddd[ii]
    for ii in range(len(ddg)):
        ddg[ii] = ddg[ii] / len(cur_time)
    ng.append(ddg)

    # iterate to find max element
    for f in ng:
        for ff in f:
            maxx = max(ff, maxx)

    # now take logarithm of everything
    for i in range(len(ng)):
        for j in range(len(ng[i])):
            ng[i][j] = ng[i][j] / maxx
            ng[i][j] = max(ng[i][j], 0.00000000000001)
            # we take minus to ensure we have all positive numbers (and one zero)
            ng[i][j] = -log(ng[i][j], 2)
            # make it pan out at something
            ng[i][j] = min(ng[i][j], 15)
            ng[i][j] = max(ng[i][j], 0)

    # we now find max once again
    # by now, we will have a logarithmic scale from 0 to 1 (which is inverted!)
    maxx = 0
    for f in ng:
        for ff in f:
            maxx = max(ff, maxx)

    for i in range(len(ng)):
        for j in range(len(ng[i])):
            ng[i][j] = ng[i][j] / maxx
            ng[i][j] = 1 - ng[i][j]
            assert 0 <= ng[i][j] and ng[i][j] <= 1

    ns = np.array(ng)
#    print(ns)

#    print("shape of ns: " + str(np.shape(ns)));


    
    # testing
    """g = s[13,:]
    print("shg:" + str(np.shape(g)))

    freqArray = np.arange(0, nUniquePts, 1.0) * (samplerate / n)
    plt.plot(freqArray/1000, 10*np.log10(g), color='k')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power (dB)')
    plt.show()"""

    return ns


def plot_spec(adp):
    ns = spectrogram(adp,time_compression=10,frequency_compression=10,cutoff=350)
    # purple means close to 0, yellow means close to 1
    plt.pcolormesh(np.transpose(ns))
    print("shape of ns: " + str(np.shape(ns)))
    plt.savefig(adp[:-4] + ".png")
