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


def spectrogram(audiopath):
    binsize = 2**11
    samplerate, samples = wav.read(audiopath)
    print(len(samples))
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
    print("shape of s: " + str(np.shape(s)));

    ng = []
    compression = 10
    cutoff = 300
    for r in s:
        i = 0
        rg = []
        while i < cutoff:
            avg = 0
            ni = i
            while ni < len(r) and ni - i < 5:
                avg += r[ni]
                ni += 1
            avg /= (ni-i)
            i = ni
            rg.append(avg)
        ng.append(rg)

    ns = np.array(ng)

    print("shape of ns: " + str(np.shape(ns)));


    
    # testing
    """g = s[13,:]
    print("shg:" + str(np.shape(g)))

    freqArray = np.arange(0, nUniquePts, 1.0) * (samplerate / n)
    plt.plot(freqArray/1000, 10*np.log10(g), color='k')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power (dB)')
    plt.show()"""

    return ns


    
spectrogram("../data/tmpvoices/skanska/fivesecfiles/out001005.wav")
spectrogram("../data/tmpvoices/skanska/fivesecfiles/out001000.wav")
#plotstft("../data/tmpvoices/skanska/fivesecfiles/out001002.wav", plotpath="skanska11.pdf", binsize=2**10)
#plotstft("440_sine.wav")
#plotstft("out001000test.wav", plotpath="skanska1.pdf")
