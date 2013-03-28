"""Functions that generate mock data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd
from nose import with_setup
import shutil


# -----------------------------------------------------------------------------
# Data creation methods
# -----------------------------------------------------------------------------
def create_waveforms(nspikes, nsamples, nchannels):
    return np.array(rnd.randint(size=(nspikes, nsamples, nchannels),
        low=-32768, high=32768), dtype=np.int16)
    
def create_features(nspikes, nchannels, fetdim):
    return np.array(rnd.randint(size=(nspikes, nchannels * fetdim + 1),
        low=-32768, high=32768), dtype=np.int16)
    
def create_clusters(nspikes, nclusters):
    return rnd.randint(size=nspikes, low=0, high=nclusters)
    
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
    return np.random.randint(size=(nchannels, 2), low=0, high=10)
