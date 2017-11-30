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

def hej(audiopath):
    samplerate, samples = wav.read(audiopath)
    binsize = 2**20
    s = stft(samples, binsize)

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
    print("frames shape: " + str(np.shape(frames)))
    frames *= win
    
    return np.fft.rfft(frames)    
    
""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)
    print("orig timebins: " + str(timebins))
    print("orig freqbins: " + str(freqbins))

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]
    
    return newspec, freqs


""" plot spectrogram"""
def plotstft(audiopath, binsize=2**10, plotpath=None, colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    print("shape of wav samples: " + str(np.shape(samples)))
    s = stft(samples, binsize)

    print("timebins: " + str(np.shape(s)[0]))
    print("freqbins: " + str(np.shape(s)[1]))

    n = binsize
    nUniquePts = int(ceil((n+1)/2.0))
    s = abs(s)

    s = s / float(n)
    s = s**2

    #s[:,arange(1,len(s)-1)] = s[:,arange(1, len(s) - 1))
    s[:,1:len(s)-1] = s[:,1:len(s)-1]*2



    # testing
    g = s[100,:]
    print("shg:" + str(np.shape(g)))

    freqArray = np.arange(0, nUniquePts, 1.0) * (samplerate / n)
    plt.plot(freqArray/1000, 10*np.log10(g), color='k')
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power (dB)')
    plt.show()


    

plotstft("../data/tmpvoices/skanska/fivesecfiles/out001002.wav", plotpath="skanska11.pdf", binsize=2**10)
#plotstft("out001000test.wav", plotpath="skanska1.pdf")
        
