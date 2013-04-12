"""Functions that generate mock data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd
import shutil
from nose import with_setup

from spiky.colors import COLORS_COUNT


# -----------------------------------------------------------------------------
# Data creation methods
# -----------------------------------------------------------------------------
def create_waveforms(nspikes, nsamples, nchannels):
    t = np.linspace(-np.pi, np.pi, nsamples)
    t = t.reshape((1, -1, 1))
    return (np.array(rnd.randint(size=(nspikes, nsamples, nchannels),
        low=-32768 // 5, high=32768 // 5), dtype=np.int16) +
            np.array((32768 * 4) // 5 * np.cos(t), dtype=np.int16))
    
def create_features(nspikes, nchannels, fetdim):
    return np.array(rnd.randint(size=(nspikes, nchannels * fetdim + 1),
        low=-32768, high=32768), dtype=np.int16)
    
def create_clusters(nspikes, nclusters):
    return rnd.randint(size=nspikes, low=0, high=nclusters)
    
def create_cluster_colors(maxcluster):
    return np.mod(np.arange(maxcluster + 1, dtype=np.int32), COLORS_COUNT) + 1
    
def create_masks(nspikes, nchannels, fetdim):
    return rnd.rand(nspikes, nchannels * fetdim + 1) < .1
    
def create_xml(nchannels, nsamples, fetdim):
    channels = '\n'.join(["<channel>{0:d}</channel>".format(i) 
        for i in xrange(nchannels)])
    xml = """
    <parameters>
      <acquisitionSystem>
        <nBits>16</nBits>
        <nChannels>{0:d}</nChannels>
        <samplingRate>20000</samplingRate>
        <voltageRange>20</voltageRange>
        <amplification>1000</amplification>
        <offset>2048</offset>
      </acquisitionSystem>
      <anatomicalDescription>
        <channelGroups>
          <group>
            {2:s}
          </group>
        </channelGroups>
      </anatomicalDescription>
      <spikeDetection>
        <channelGroups>
          <group>
            <channels>
              {2:s}
            </channels>
            <nSamples>{1:d}</nSamples>
            <peakSampleIndex>10</peakSampleIndex>
            <nFeatures>{3:d}</nFeatures>
          </group>
        </channelGroups>
      </spikeDetection>
    </parameters>
    """.format(nchannels, nsamples, channels, fetdim)
    return xml

def create_probe(nchannels):
    # return np.random.randint(size=(nchannels, 2), low=0, high=10)
    probe = np.zeros((nchannels, 2), dtype=np.int32)
    probe[:, 0] = np.arange(nchannels)
    probe[::2, 0] *= -1
    probe[:, 1] = np.arange(nchannels)
    return probe
