"""Functions that generate mock data."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os

import numpy as np
import numpy.random as rnd
import pandas as pd
import shutil

from spiky.utils.colors import COLORS_COUNT
from spiky.io.tools import save_binary, save_text, check_dtype, check_shape
from spiky.stats.cache import IndexedMatrix


# -----------------------------------------------------------------------------
# Global variables
# -----------------------------------------------------------------------------
# Mock parameters.
nspikes = 1000
nclusters = 20
cluster_offset = 2
nsamples = 20
ncorrbins = 50
nchannels = 32
fetdim = 3


# -----------------------------------------------------------------------------
# Data creation methods
# -----------------------------------------------------------------------------
def create_waveforms(nspikes, nsamples, nchannels):
    t = np.linspace(-np.pi, np.pi, nsamples)
    t = t.reshape((1, -1, 1))
    # Sinus shaped random waveforms.
    return (np.array(rnd.randint(size=(nspikes, nsamples, nchannels),
        low=-32768 // 2, high=32768 // 2), dtype=np.int16) -
            np.array(32768 // 2 * (.5 + .5 * rnd.rand()) * np.cos(t),
            dtype=np.int16))
    
def create_features(nspikes, nchannels, fetdim):
    return np.array(rnd.randint(size=(nspikes, nchannels * fetdim + 1),
        low=-32768, high=32768), dtype=np.int16)
    
def create_clusters(nspikes, nclusters):
    # Add shift in cluster indices to test robustness.
    return rnd.randint(size=nspikes, low=cluster_offset, 
        high=nclusters + cluster_offset)
    
def create_cluster_colors(maxcluster):
    return np.mod(np.arange(maxcluster + 1, dtype=np.int32), COLORS_COUNT) + 1
    
def create_masks(nspikes, nchannels, fetdim):
    return rnd.rand(nspikes, nchannels * fetdim + 1) < .1
    
def create_correlation_matrix(nclusters):
    return np.random.rand(nclusters, nclusters)
    
def create_correlograms(clusters, ncorrbins):
    n = len(clusters)
    shape = (n, n, ncorrbins)
    return IndexedMatrix(clusters, shape=shape,
        data=np.random.rand(*shape))
    
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

    
# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
def setup():
    # Create mock data.
    waveforms = create_waveforms(nspikes, nsamples, nchannels)
    features = create_features(nspikes, nchannels, fetdim)
    clusters = create_clusters(nspikes, nclusters)
    cluster_colors = create_cluster_colors(nclusters - 1)
    masks = create_masks(nspikes, nchannels, fetdim)
    xml = create_xml(nchannels, nsamples, fetdim)
    probe = create_probe(nchannels)
    
    # Create mock directory if needed.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    # Create mock files.
    save_binary(os.path.join(dir, 'test.spk.1'), waveforms)
    save_text(os.path.join(dir, 'test.fet.1'), features,
        header=nchannels * fetdim + 1)
    save_text(os.path.join(dir, 'test.clu.1'), clusters, header=nclusters)
    # save_text(os.path.join(dir, 'test.clucol.1'), cluster_colors)
    save_text(os.path.join(dir, 'test.mask.1'), masks, header=nclusters)
    save_text(os.path.join(dir, 'test.xml'), xml)
    save_text(os.path.join(dir, 'test.probe'), probe)
    
def teardown():
    # Erase the temporary data directory.
    dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mockdata')
    if os.path.exists(dir):
        shutil.rmtree(dir)
    
