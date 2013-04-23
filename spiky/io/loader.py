"""This module provides utility classes and functions to load spike sorting
data sets."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os.path
import re
from collections import Counter

import numpy as np
import pandas as pd

from tools import (find_filename, find_index, load_text, load_xml, normalize,
    load_binary, load_pickle)
from selection import (select, select_pairs, get_spikes_in_clusters,
    get_some_spikes_in_clusters)
from spiky.utils.userpref import USERPREF
from spiky.utils.logger import debug, info, warn
from spiky.utils.colors import COLORS_COUNT


# -----------------------------------------------------------------------------
# File loading functions
# -----------------------------------------------------------------------------
def read_xml(filename_xml, fileindex):
    """Read the XML file associated to the current dataset,
    and return a metadata dictionary."""
    
    params = load_xml(filename_xml, fileindex=fileindex)
    
    # klusters tests
    metadata = dict(
        nchannels=params['nchannels'],
        nsamples=params['nsamples'],
        fetdim=params['fetdim'],
        freq=params['rate'])
    
    return metadata

def read_clusters_info(filename_cluinfo, fileindex):
    info = load_pickle(filename_cluinfo)
    return info
    
def read_features(filename_fet, nchannels, fetdim, freq):
    """Read a .fet file and return the normalize features array,
    as well as the spiketimes."""
    
    features = load_text(filename_fet, np.int32, skiprows=1)
    features = np.array(features, dtype=np.float32)
    
    # HACK: There are either 1 or 5 dimensions more than fetdim*nchannels
    # we can't be sure so we first try 1, if it does not work we try 5.
    for nextrafet in [1, 5]:
        try:
            features = features.reshape((-1,
                                         fetdim * nchannels + nextrafet))
            # if the features array could be reshape, directly break the loop
            break
        except ValueError:
            features = None
    if features is None:
        raise ValueError("""The number of columns in the feature matrix
        is not fetdim (%d) x nchannels (%d) + 1 or 5.""" % 
            (fetdim, nchannels))
    
    # get the spiketimes
    spiketimes = features[:,-1].copy()
    spiketimes *= (1. / freq)
    
    # count the number of extra features
    nextrafet = features.shape[1] - nchannels * fetdim
    
    # normalize normal features while keeping symmetry
    features[:,:-nextrafet] = normalize(features[:,:-nextrafet],
                                        symmetric=True)
    # normalize extra features without keeping symmetry
    features[:,-nextrafet:] = normalize(features[:,-nextrafet:],
                                        symmetric=False)
    
    return features, spiketimes
    
def read_clusters(filename_clu):
    clusters = load_text(filename_clu, np.int32)
    clusters = clusters[1:]
    return clusters

def read_cluster_info(filename_clusterinfo):
    # For each cluster (absolute indexing): color index, and group index
    cluster_info = load_text(filename_clusterinfo, np.int32)
    return cluster_info
    
def read_group_info(filename_groups):
    # For each group (absolute indexing): color index, and name
    group_info = load_text(filename_groups, str)
    return group_info
    
def read_masks(filename_mask, fetdim):
    masks_full = load_text(filename_mask, np.float32, skiprows=1)
    masks = masks_full[:,:-1:fetdim]
    return masks, masks_full
    
def read_waveforms(filename_spk, nsamples, nchannels):
    waveforms = np.array(load_binary(filename_spk), dtype=np.float32)
    waveforms = normalize(waveforms)
    waveforms = waveforms.reshape((-1, nsamples, nchannels))
    return waveforms

def read_probe(filename_probe):
    return normalize(np.array(load_text(filename_probe, np.int32),
        dtype=np.float32))


# -----------------------------------------------------------------------------
# Generic Loader class
# -----------------------------------------------------------------------------
class Loader(object):
    
    # Initialization methods
    # ----------------------
    def __init__(self, filename=None):
        """Initialize a Loader object for loading Klusters-formatted files.
        
        Arguments:
          * filename: the full path of any file belonging to the same
            dataset.
        
        """
        self.spikes_selected = None
        self.clusters_selected = None
        if filename:
            self.open(filename)
    
    def open(self, filename):
        pass
    
    
    # Input-Output methods
    # --------------------
    def read(self):
        pass
    
    def save(self):
        pass
    
    
    # Access to the data: spikes
    # --------------------------
    def select(self, spikes=None, clusters=None):
        pass
    
    def get_clusters_selected(self):
        return self.clusters_selected
    
    def get_features(self, spikes=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)
        if spikes is None:
            spikes = self.spikes_selected
        return select(self.features, spikes)
    
    def get_spiketimes(self, spikes=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)
        if spikes is None:
            spikes = self.spikes_selected
        return select(self.spiketimes, spikes)
    
    def get_clusters(self, spikes=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)
        if spikes is None:
            spikes = self.spikes_selected
        return select(self.clusters, spikes)
    
    def get_masks(self, spikes=None, full=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)
        if spikes is None:
            spikes = self.spikes_selected
        if not full:
            masks = self.masks
        else:
            masks = self.masks_full
        return select(masks, spikes)
    
    def get_waveforms(self, spikes=None, clusters=None):
        if spikes is not None:
            return select(self.waveforms, spikes)
        else:
            if clusters is None:
                clusters = self.clusters_selected
            if clusters is not None:
                spikes = get_some_spikes_in_clusters(clusters, self.clusters,
                    nspikes_max_expected=USERPREF['waveforms_nspikes_max_expected'],
                    nspikes_per_cluster_min=USERPREF['waveforms_nspikes_per_cluster_min'])
            else:
                spikes = self.spikes_selected
        return select(self.waveforms, spikes)
    
    
    # Access to the data: clusters
    # ----------------------------
    def get_cluster_colors(self, clusters=None):
        if clusters is None:
            clusters = self.clusters_selected
        return select(self.cluster_colors, clusters)
    
    def get_cluster_groups(self, clusters=None):
        if clusters is None:
            clusters = self.clusters_selected
        return select(self.cluster_groups, clusters)
    
    def get_group_colors(self, groups=None):
        return select(self.group_colors, groups)
    
    def get_group_names(self, groups=None):
        return select(self.group_names, groups)
    
    def get_cluster_sizes(self, clusters=None):
        if clusters is None:
            clusters = self.clusters_selected
        counter = Counter(self.clusters)
        sizes = pd.Series(counter, dtype=np.int32)
        return select(sizes, clusters)
    
    
    # Access to the data: stats
    # -------------------------
    # def get_correlograms(self, clusters=None):
        # if clusters is None:
            # clusters = self.clusters_selected
        # return select_pairs(self.correlograms, clusters)
        
    # def get_correlation_matrix(self):
        # return self.correlation_matrix
        
        
    # Access to the data: misc
    # ------------------------
    def get_probe(self):
        return self.probe
    
    
    # Setters
    # -------
    # def set_correlograms(self, correlograms):
        # self.correlograms.update(correlograms)
        
    # def set_correlation_matrix(self, correlation_matrix):
        # self.correlation_matrix = correlation_matrix
        
    # def invalidate_correlograms(self, cluster_indices):
        # pass

    
# -----------------------------------------------------------------------------
# Klusters Loader
# -----------------------------------------------------------------------------
class KlustersLoader(Loader):
    
    def open(self, filename):
        """Open a file."""
        self.filename = filename
        # Find the file index associated to the filename, or 1 by default.
        self.fileindex = find_index(filename) or 1
        self.find_filenames()
        self.read()
        
    def find_filenames(self):
        """Find the filenames of the different files for the current
        dataset."""
        self.filename_xml = find_filename(self.filename, 'xml')
        self.filename_fet = find_filename(self.filename, 'fet')
        self.filename_clu = find_filename(self.filename, 'clu')
        self.filename_clusterinfo = find_filename(self.filename, 'clusterinfo')
        self.filename_groups = find_filename(self.filename, 'groups')
        # fmask or mask file
        self.filename_mask = find_filename(self.filename, 'fmask')
        if not self.filename_mask:
            self.filename_mask = find_filename(self.filename, 'mask')
        self.filename_spk = find_filename(self.filename, 'spk')
        self.filename_probe = find_filename(self.filename, 'probe')
    
    # Input-Output methods
    # --------------------
    def read(self):
        info("Opening {0:s}.".format(self.filename))
        
        # Read metadata.
        # --------------
        try:
            self.metadata = read_xml(self.filename_xml, self.fileindex)
        except IOError:
            # Die if no XML file is available for this dataset, as it contains
            # critical metadata.
            raise IOError("The XML file is missing.")
            
        nsamples = self.metadata.get('nsamples')
        nchannels = self.metadata.get('nchannels')
        fetdim = self.metadata.get('fetdim')
        freq = self.metadata.get('freq')
        
        # Read probe.
        # -----------
        try:
            self.probe = read_probe(self.filename_probe)
        except IOError:
            self.probe = None
        
        
        # Read features.
        # --------------
        try:
            self.features, self.spiketimes = read_features(self.filename_fet,
                nchannels, fetdim, freq)
        except IOError:
            raise IOError("The FET file is missing.")
        # Convert to Pandas.
        self.features = pd.DataFrame(self.features, dtype=np.float32)
        self.spiketimes = pd.Series(self.spiketimes, dtype=np.float32)
        
        # Count the number of spikes and save it in the metadata.
        nspikes = self.features.shape[0]
        self.metadata['nspikes'] = nspikes
        
        # Read clusters.
        # --------------
        try:
            self.clusters = read_clusters(self.filename_clu)
        except IOError:
            warn("The CLU file is missing.")
            # Default clusters if the CLU file is not available.
            self.clusters = np.zeros(nspikes + 1, dtype=np.int32)
            self.clusters[0] = 1
        # Convert to Pandas.
        self.clusters = pd.Series(self.clusters, dtype=np.int32)
        
        # Counter clusters.
        counter = Counter(self.clusters)
        self.nclusters = len(counter)
        clusters_unique = sorted(counter.keys())
        
        # Read cluster info.
        # ------------------
        try:
            self.cluster_info = read_cluster_info(self.filename_clusterinfo)
        except IOError:
            info("The CLUSTERINFO file is missing.")
            maxcluster = max(clusters_unique)
            self.cluster_info = np.zeros((maxcluster + 1, 2), dtype=np.int32)
            self.cluster_info[:, 0] = np.mod(np.arange(maxcluster + 1, 
                dtype=np.int32), COLORS_COUNT) + 1
            # First column: color index, second column: group index (2 by
            # default)
            self.cluster_info[:, 1] = 2 * np.ones(maxcluster + 1)
        # Convert to Pandas.
        self.cluster_info = pd.DataFrame(self.cluster_info, dtype=np.int32)
        self.cluster_info = select(self.cluster_info, clusters_unique)
        self.cluster_colors = self.cluster_info[0].astype(np.int32)
        self.cluster_groups = self.cluster_info[1].astype(np.int32)
        
        # Read group info.
        # ----------------
        try:
            self.group_info = read_group_info(self.filename_groups)
        except IOError:
            info("The GROUPS file is missing.")
            self.group_info = np.zeros((3, 2), dtype=object)
            self.group_info[:,0] = (#np.array(
                np.mod(np.arange(3), COLORS_COUNT) + 1)#, dtype=str)
            self.group_info[:,1] = np.array(['Noise', 'MUA', 'Good'],
                dtype=object)
        # Convert to Pandas.
        self.group_info = pd.DataFrame(self.group_info)
        self.group_colors = self.group_info[0].astype(np.int32)
        self.group_names = self.group_info[1].astype(np.str_)
        
        # Read masks.
        # -----------
        try:
            self.masks, self.masks_full = read_masks(self.filename_mask,
                                                     fetdim)
        except IOError:
            warn("The MASKS/FMASKS file is missing.")
            # Default masks if the MASK/FMASK file is not available.
            self.masks = np.ones((nspikes, nchannels))
            self.masks_full = np.ones(features.shape)
        self.masks = pd.DataFrame(self.masks)
        self.masks_full = pd.DataFrame(self.masks_full)

        # Read waveforms.
        # ---------------
        try:
            self.waveforms = read_waveforms(self.filename_spk, nsamples,
                                            nchannels)
        except IOError:
            warn("The SPK file is missing.")
            self.waveforms = np.zeros((nspikes, nsamples, nchannels))
        # Convert to Pandas.
        self.waveforms = pd.Panel(self.waveforms, dtype=np.float32)
    
        # Initialize cluster statistics
        # -----------------------------
        # TODO: compute them in an external process
        cluster_max = self.clusters.max()
        self.ncorrbins = 50
        self.corrbin = .001
        # self.correlograms = {}
        # self.correlation_matrix = np.zeros((self.nclusters, self.nclusters))
    
        # Save data set parameters.
        self.nsamples = nsamples
        self.nchannels = nchannels
        self.fetdim = fetdim
        self.freq = freq
        self.nextrafet = self.features.shape[1] - nchannels * fetdim
    
    def save(self):
        pass
    
    # def close(self):
        # self.spikes_selected = None
        # self.clusters_selected = None
        
        # self.filename = None
        # self.fileindex = None
        # self.filename_xml = None
        # self.filename_fet = None
        # self.filename_clu = None
        # self.filename_mask = None
        # self.filename_spk = None
        
        # self.features = None
        # self.spiketimes = None
        # self.clusters = None
        # self.masks = None
        # self.masks_full = None
        # self.waveforms = None
        
        # self.metadata = {}
    
    def select(self, spikes=None, clusters=None):
        if clusters is not None:
            spikes = get_spikes_in_clusters(clusters, self.clusters)    
        self.spikes_selected = spikes
        self.clusters_selected = clusters
