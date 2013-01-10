import numpy as np
import numpy.random as rdn
import collections
import operator
import time

from galry import *
from common import HighlightManager, SpikyBindings, SpikeDataOrganizer
from widgets import VisualizationWidget
import spiky.tools as stools
import spiky.colors as scolors
import spiky.signals as ssignals

__all__ = ['WaveformView', 'WaveformWidget']

FSH = """
vec3 Hue(float H)
{
    float R = abs(H * 6 - 3) - 1;
    float G = 2 - abs(H * 6 - 2);
    float B = 2 - abs(H * 6 - 4);
    return vec3(clamp(R, 0, 1), clamp(G, 0, 1), clamp(B, 0, 1));
}

vec3 HSVtoRGB(vec3 HSV)
{
    return ((Hue(HSV.x) - 1) * HSV.y + 1) * HSV.z;
}

vec3 RGBtoHSV(vec3 RGB)
{
    vec3 HSV = vec3(0, 0, 0);
    HSV.z = max(RGB.r, max(RGB.g, RGB.b));
    float M = min(RGB.r, min(RGB.g, RGB.b));
    float C = HSV.z - M;
    if (C != 0)
    {
        HSV.y = C / HSV.z;
        vec3 Delta = (HSV.z - RGB) / C;
        Delta.rgb -= Delta.brg;
        Delta.rg += vec2(2,4);
        if (RGB.r >= HSV.z)
            HSV.x = Delta.b;
        else if (RGB.g >= HSV.z)
            HSV.x = Delta.r;
        else
            HSV.x = Delta.g;
        HSV.x = fract(HSV.x / 6);
    }
    return HSV;
}
"""


VERTEX_SHADER = """
    // get channel position
    vec2 channel_position = channel_positions[int(channel)];
    
    // get the box position
    vec2 box_position = channel_position;
    
    // take probe scaling into account
    box_position *= probe_scale;
    
    // adjust box position in the separated case
    if (!superimposed)
    {
        box_position.x += box_size_margin.x * (0.5 + cluster - 0.5 * nclusters);
    }
    
    // compute the depth: put masked spikes on the background, unmasked ones
    // on the foreground on a different layer for each cluster
    float depth = 0.;
    if (mask == 1.)
        depth = -(cluster + 1) / nclusters;
    
    // move the vertex to its position0
    vec3 position = vec3(position0 * 0.5 * box_size + box_position, depth);
    
    vhighlight = highlight;
    cmap_vindex = cmap_index;
    vmask = mask;
    
"""
        
FRAGMENT_SHADER = """
    float index = %CMAP_OFFSET% + cmap_vindex * %CMAP_STEP%;
    out_color = texture1D(cmap, index);
    
    // TODO
    if (vmask == 0) {
        out_color.xyz = vec3(.5, .5, .5);
    }
    out_color.w = .25 + .75 * vmask;
        
    if (vhighlight > 0)  {
        //out_color.xyz = out_color.xyz + vec3(.5, .5, .5);
        vec3 hsv = RGBtoHSV(out_color.xyz);
        hsv.y = clamp(hsv.y - .5, 0, 1);
        hsv.z = clamp(hsv.z + .5, 0, 1);
        out_color.xyz = HSVtoRGB(hsv);
    }
    
    //out_color.w = 1;
"""


class WaveformHighlightManager(HighlightManager):
    def initialize(self):
        """Set info from the data manager."""
        super(WaveformHighlightManager, self).initialize()
        data_manager = self.data_manager
        # self.get_data_position = self.data_manager.get_data_position
        self.full_masks = self.data_manager.full_masks
        self.clusters_rel = self.data_manager.clusters_rel
        self.cluster_colors = self.data_manager.cluster_colors
        self.nchannels = data_manager.nchannels
        self.nclusters = data_manager.nclusters
        self.nsamples = data_manager.nsamples
        self.nspikes = data_manager.nspikes
        self.npoints = data_manager.npoints
        # self.get_data_position = data_manager.get_data_position
        self.highlighted_spikes = []
        self.highlight_mask = np.zeros(self.npoints, dtype=np.int32)
        self.highlighting = False
        
    def find_enclosed_spikes(self, enclosing_box):
        
        if self.nspikes == 0:
            return np.array([])
        
        # first call
        if not self.highlighting:
            # create
            box_positions, box_size = self.position_manager.get_transformation()
            # Tx, Ty: Nchannels x Nclusters
            Tx, Ty = box_positions
            # to: Nspikes x Nsamples x Nchannels
            Px = np.tile(Tx[:,self.clusters_rel].reshape((self.nchannels, self.nspikes, 1)), (1, 1, self.nsamples))
            self.Px = Px.transpose([1, 2, 0])
            Py = np.tile(Ty[:,self.clusters_rel].reshape((self.nchannels, self.nspikes, 1)), (1, 1, self.nsamples))
            self.Py = Py.transpose([1, 2, 0])
            
            self.Wx = np.tile(np.linspace(-1., 1., self.nsamples).reshape((1, -1, 1)), (self.nspikes, 1, self.nchannels))
            self.Wy = self.data_manager.waveforms_reordered
            
            self.highlighting = True
        
        x0, y0, x1, y1 = enclosing_box
        
        # press_position
        xp, yp = x0, y0

        # reorder
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)

        # transformation
        box_positions, box_size = self.position_manager.get_transformation()
        Tx, Ty = box_positions
        w, h = box_size
        a, b = w / 2, h / 2
        
        # find the enclosed channels and clusters
        channels, clusters = self.position_manager.get_enclosed_channels((x0, y0, x1, y1))
        channels = np.unique(channels)
        clusters = np.unique(clusters)
        
        # print channels
        
        if channels.size == 0:
            return np.array([])
        
        u, v = self.Px[:,:,channels], self.Py[:,:,channels]
        Wx, Wy = self.Wx[:,:,channels], self.Wy[:,:,channels]
        
        ind =  ((Wx >= (xmin-u)/a) & (Wx <= (xmax-u)/a) & \
                (Wy >= (ymin-v)/b) & (Wy <= (ymax-v)/b))
        
        spkindices = np.nonzero(ind.max(axis=1).max(axis=1))[0]

        return spkindices

    def find_indices_from_spikes(self, spikes):
        if spikes is None or len(spikes)==0:
            return None
        n = len(spikes)
        # find point indices in the data buffer corresponding to 
        # the selected spikes. In particular, waveforms of those spikes
        # across all channels should be selected as well.
        spikes = np.array(spikes, dtype=np.int32)
        ind = np.repeat(spikes * self.nsamples, self.nsamples)
        ind += np.tile(np.arange(self.nsamples), n)
        ind = np.tile(ind, self.nchannels)
        ind += np.repeat(np.arange(self.nchannels) * self.nsamples * self.nspikes,
                                self.nsamples * n)
        return ind
        
    def set_highlighted_spikes(self, spikes, do_emit=True):
        """Update spike colors to mark transiently selected spikes with
        a special color."""
        if len(spikes) == 0:
            # do update only if there were previously selected spikes
            do_update = len(self.highlighted_spikes) > 0
            self.highlight_mask[:] = 0
        else:
            do_update = True
            ind = self.find_indices_from_spikes(spikes)
        
            self.highlight_mask[:] = 0
            self.highlight_mask[ind] = 1
        
        if do_update:
            
            # emit the HighlightSpikes signal
            if do_emit:
                ssignals.emit(self.parent, 'HighlightSpikes', spikes)
                
            self.paint_manager.set_data(
                highlight=self.highlight_mask,
                visual='waveforms')
        
        self.highlighted_spikes = spikes
        
    def highlighted(self, box):
        # get selected spikes
        spikes = self.find_enclosed_spikes(box) 
        
        # update the data buffer
        self.set_highlighted_spikes(spikes)
    
    def cancel_highlight(self):
        super(WaveformHighlightManager, self).cancel_highlight()
        self.set_highlighted_spikes(np.array([]))
        self.highlighting = False
    
    
class WaveformPositionManager(Manager):
    # Initialization methods
    # ----------------------
    def reset(self):
        # set parameters
        self.alpha = .02
        self.beta = .02
        self.box_size_min = .01
        self.probe_scale_min = .01
        self.probe_scale = (1., 1.)
        # for each spatial arrangement, the box sizes automatically computed,
        # or modified by the user
        self.box_sizes = dict()
        self.box_sizes.__setitem__('Linear', None)
        self.box_sizes.__setitem__('Geometrical', None)
        # self.T = None
        self.spatial_arrangement = 'Linear'
        self.superposition = 'Separated'
        
        # channel positions
        self.channel_positions = {}
        
    def normalize_channel_positions(self, spatial_arrangement, channel_positions):
        channel_positions = channel_positions.copy()
        
        # waveform data bounds
        xmin = channel_positions[:,0].min()
        xmax = channel_positions[:,0].max()
        ymin = channel_positions[:,1].min()
        ymax = channel_positions[:,1].max()
        
        w, h = self.find_box_size(spatial_arrangement=spatial_arrangement)
    
        # HACK: if the normalization depends on the number of clusters,
        # the positions will change whenever the cluster selection changes
        # k = self.nclusters
        k = 3
        
        if xmin == xmax:
            ax = 0.
        else:
            ax = (2 - k * w * (1 + 2 * self.alpha)) / (xmax - xmin)
        if ymin == ymax:
            ay = 0.
        else:
            ay = (2 - h * (1 + 2 * self.alpha)) / (ymax - ymin)
        
        # set bx and by to have symmetry
        bx = -.5 * ax * (xmax + xmin)
        by = -.5 * ay * (ymax + ymin)
        
        # transform the boxes positions so that everything fits on the screen
        channel_positions[:,0] = ax * channel_positions[:,0] + bx
        channel_positions[:,1] = ay * channel_positions[:,1] + by
        
        return enforce_dtype(channel_positions, np.float32)
    
    def get_channel_positions(self):
        return self.channel_positions[self.spatial_arrangement]
    
    def set_info(self, nchannels, nclusters, 
                       geometrical_positions=None,
                       spatial_arrangement=None, superposition=None,
                       box_size=None, probe_scale=None):
        """Specify the information needed to position the waveforms in the
        widget.
        
          * nchannels: number of channels
          * nclusters: number of clusters
          * coordinates of the electrodes
          
        """
        self.nchannels = nchannels
        self.nclusters = nclusters
        
        # HEURISTIC
        # self.diffxc, self.diffyc = [np.sqrt(float(self.nchannels))] * 2
        
        linear_positions = np.zeros((self.nchannels, 2), dtype=np.float32)
        linear_positions[:,1] = np.linspace(-1., 1., self.nchannels)
        
        # default geometrical position
        if geometrical_positions is None:
            geometrical_positions = linear_positions.copy()
                         
        # normalize and save channel position
        self.channel_positions['Linear'] = \
            self.normalize_channel_positions('Linear', linear_positions)
        self.channel_positions['Geometrical'] = \
            self.normalize_channel_positions('Geometrical', geometrical_positions)
              
        
        # set waveform positions
        self.update_arrangement(spatial_arrangement=spatial_arrangement,
                                superposition=superposition,
                                box_size=box_size,
                                probe_scale=probe_scale)
        
    def update_arrangement(self, spatial_arrangement=None, superposition=None,
                                 box_size=None, probe_scale=None):
        """Update the waveform arrangement (self.channel_positions).
        
          * spatial_arrangement: 'Linear' or 'Geometrical'
          * superposition: 'Superimposed' or 'Separated'
          
        """
        # save spatial arrangement
        if spatial_arrangement is not None:
            self.spatial_arrangement = spatial_arrangement
        if superposition is not None:
            self.superposition = superposition
        
        # save box size
        if box_size is not None:
            self.save_box_size(*box_size)
        
        # save probe scale
        if probe_scale is not None:
            self.probe_scale = probe_scale
        
        # retrieve info
        channel_positions = self.channel_positions[self.spatial_arrangement]
        
        size = self.load_box_size()
        if size is None:
            w = h = 0.
        else:
            w, h = size
        
        # update translation vector
        # order: cluster, channel
        T = np.repeat(channel_positions, self.nclusters, axis=0)
        Tx = np.reshape(T[:,0], (self.nchannels, self.nclusters))
        Ty = np.reshape(T[:,1], (self.nchannels, self.nclusters))
        
        # take probe scale into account
        psx, psy = self.probe_scale
        Tx *= psx
        Ty *= psy
        
        # shift in the separated case
        if self.superposition == 'Separated':
            clusters = np.tile(np.arange(self.nclusters), (self.nchannels, 1))
            Tx += w * (1 + 2 * self.alpha) * \
                                    (.5 + clusters - self.nclusters / 2.)

        # record box positions and size
        self.box_positions = Tx, Ty
        self.box_size = (w, h)
                      
    def get_transformation(self):
        return self.box_positions, self.box_size


    # Internal methods
    # ----------------
    def save_box_size(self, w, h, arrangement=None):
        if arrangement is None:
            arrangement = self.spatial_arrangement
        self.box_sizes[arrangement] = (w, h)

    def load_box_size(self, arrangement=None):
        if arrangement is None:
            arrangement = self.spatial_arrangement
        if self.box_sizes[arrangement] is None:
            self.find_box_size()
        return self.box_sizes[arrangement]
    
    def find_box_size(self, spatial_arrangement=None, superposition=None):
        if self.nclusters == 0:
            return 0., 0.
        do_save = (spatial_arrangement is None) and (superposition is None)
        
        w = .5
        h = .1
        
        if do_save:
            self.save_box_size(w, h)
        return w, h
        
        
    # Interactive update methods
    # --------------------------
    def change_box_scale(self, dsx, dsy):
        w, h = self.load_box_size()
        w = max(self.box_size_min, w + dsx)
        h = max(self.box_size_min, h + dsy)
        self.update_arrangement(box_size=(w,h))
        self.paint_manager.auto_update_uniforms("box_size", "box_size_margin")
        
    def change_probe_scale(self, dsx, dsy):
        # w, h = self.load_box_size()
        sx, sy = self.probe_scale
        sx = max(self.probe_scale_min, sx + dsx)
        sy = max(self.probe_scale_min, sy + dsy)
        self.update_arrangement(probe_scale=(sx, sy))
        self.paint_manager.auto_update_uniforms("probe_scale")
        
    def toggle_superposition(self):
        # switch superposition
        if self.superposition == 'Separated':
            self.superposition = 'Superimposed'
        else:
            self.superposition = 'Separated'
        # recompute the waveforms positions
        self.update_arrangement(superposition=self.superposition,
                                spatial_arrangement=self.spatial_arrangement)
        self.paint_manager.auto_update_uniforms("superimposed", "box_size", "box_size_margin")

    def toggle_spatial_arrangement(self):
        # switch spatial arrangement
        if self.spatial_arrangement == 'Linear':
            self.spatial_arrangement = 'Geometrical'
        else:
            self.spatial_arrangement = 'Linear'
        # recompute the waveforms positions
        self.update_arrangement(superposition=self.superposition,
                                spatial_arrangement=self.spatial_arrangement)
        self.paint_manager.auto_update_uniforms("channel_positions", "box_size", "box_size_margin")
        
        
    # Get methods
    # -----------
    def get_viewbox(self, channels):
        """Return the smallest viewbox such that the selected channels are
        visible.
        """
        channels = np.array(channels)
        pos = self.box_positions[channels,:]
        # find the box enclosing all channels center positions
        xmin, ymin = np.min(pos, axis=0)
        xmax, ymax = np.max(pos, axis=0)
        # take the size of the individual boxes into account
        mx = self.w * (.5 + self.alpha)
        my = self.h * (.5 + self.alpha)
        xmin -= self.w * mx
        xmax += self.w * mx
        ymin -= self.h * my
        ymax += self.h * my
        return xmin, ymin, xmax, ymax
        
    def find_box(self, xp, yp):
        if self.nclusters == 0:
            return 0, 0
        
        # transformation
        box_positions, box_size = self.get_transformation()
        Tx, Ty = box_positions
        w, h = box_size
        a, b = w / 2, h / 2
        
        # find the enclosed channels and clusters
        sx, sy = self.interaction_manager.get_processor('navigation').sx, self.interaction_manager.get_processor('navigation').sy
        dist = (np.abs(Tx - xp) * sx) ** 2 + (np.abs(Ty - yp) * sy) ** 2
        closest = np.argmin(dist.ravel())
        channel, cluster_rel = closest // self.nclusters, np.mod(closest, self.nclusters)
        
        return channel, cluster_rel
        
    def get_enclosed_channels(self, box):
        # channel positions: Nchannels x Nclusters
        Tx, Ty = self.box_positions
        
        
        
        w, h = self.box_size
        # print w, h
        # enclosed box
        x0, y0, x1, y1 = box
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        ind = ((x0 <= Tx + w/2) & (x1 >= Tx - w/2) &
               (y0 <= Ty + h/2) & (y1 >= Ty - h/2))
        return np.nonzero(ind)


class WaveformDataManager(Manager):
    # Initialization methods
    # ----------------------
    # @profile
    def set_data(self, waveforms, clusters=None, cluster_colors=None,
                 clusters_unique=None,
                 masks=None, geometrical_positions=None, spike_ids=None,
                 spatial_arrangement=None, superposition=None,
                 box_size=None, probe_scale=None):
        """
        waveforms is a Nspikes x Nsamples x Nchannels array.
        clusters is a Nspikes array, with the cluster absolute index for each
                    spike
        cluster_colors is a Nclusters x 3 array (RGB components)
            cluster_colors[i] is the color of cluster #i where i is the RELATIVE
            index
        masks is a Nspikes x Nchannels array (with values in [0,1])
        spike_ids is a Nspikes array, it contains the absolute indices of spikes
        """
        
        # print waveforms.shape
        # print cluster_colors
        
        self.nspikes, self.nsamples, self.nchannels = waveforms.shape
        self.npoints = waveforms.size
        self.geometrical_positions = geometrical_positions
        self.spike_ids = spike_ids
        self.waveforms = waveforms
        
        # data organizer: reorder data according to clusters
        self.data_organizer = SpikeDataOrganizer(waveforms,
                                                clusters=clusters,
                                                cluster_colors=cluster_colors,
                                                clusters_unique=clusters_unique,
                                                masks=masks,
                                                nchannels=self.nchannels,
                                                spike_ids=spike_ids)
        
        # get reordered data
        # self.permutation = self.data_organizer.permutation
        self.waveforms_reordered = self.data_organizer.data_reordered
        self.nclusters = self.data_organizer.nclusters
        self.clusters = self.data_organizer.clusters
        self.masks = self.data_organizer.masks
        self.cluster_colors = self.data_organizer.cluster_colors
        self.clusters_unique = self.data_organizer.clusters_unique
        self.clusters_rel = self.data_organizer.clusters_rel
        self.cluster_sizes = self.data_organizer.cluster_sizes
        # self.cluster_sizes_cum = self.data_organizer.cluster_sizes_cum
        self.cluster_sizes_dict = self.data_organizer.cluster_sizes_dict
        
        # prepare GPU data: waveform initial positions and colors
        data = self.prepare_waveform_data()
        
        # masks
        self.full_masks = np.repeat(self.masks.T.ravel(), self.nsamples)
        self.full_clusters = np.tile(np.repeat(self.clusters_rel, self.nsamples), self.nchannels)
        self.full_channels = np.repeat(np.arange(self.nchannels, dtype=np.int32), self.nspikes * self.nsamples)
        
        # normalize the initial waveforms
        # self.data_normalizer = DataNormalizer(data)
        # self.normalized_data = self.data_normalizer.normalize()
        # NEW: normalization in dataio instead
        self.normalized_data = data
        
        # position waveforms
        self.position_manager.set_info(self.nchannels, self.nclusters, 
                                       geometrical_positions=self.geometrical_positions,
                                       spatial_arrangement=spatial_arrangement,
                                       superposition=superposition,
                                       box_size=box_size,
                                       probe_scale=probe_scale)
        
        # update the highlight manager
        self.highlight_manager.initialize()
    
    
    # Internal methods
    # ----------------
    # @profile
    def prepare_waveform_data(self):
        """Define waveform data."""
        # prepare data for GPU transfer
        # in GPU memory, X coordinates are always between -1 and 1
        X = np.tile(np.linspace(-1., 1., self.nsamples),
                                (self.nchannels * self.nspikes, 1))
        
        # waveforms_reordered: Nspikes x Nsamples x Nchannels
        # Y: (Nsamples x Nspikes) x Nchannels array
        if self.nspikes == 0:
            Y = np.array([], dtype=np.float32)
        else:
            Y = np.vstack(self.waveforms_reordered)
        
        # create a Nx2 array with all coordinates
        data = np.empty((X.size, 2), dtype=np.float32)
        data[:,0] = X.ravel()
        data[:,1] = Y.T.ravel()
        return data
    
    
class WaveformVisual(Visual):
    @staticmethod
    def get_size_bounds(nsamples=None, npoints=None):
        size = npoints
        bounds = np.arange(0, npoints + 1, nsamples)
        return size, bounds

    def initialize(self, nclusters=None, nchannels=None, 
        nsamples=None, npoints=None, #nspikes=None,
        position0=None, mask=None, cluster=None, 
        cluster_colors=None, channel=None, highlight=None):

        self.size, self.bounds = WaveformVisual.get_size_bounds(nsamples, npoints)
        
        self.primitive_type = 'LINE_STRIP'
        
        # NEW: add depth
        # depth = np.zeros((self.size, 0))
        # position = np.hstack((position0, depth))
        position = position0
        
        self.add_attribute("position0", vartype="float", ndim=2, data=position)
        
        self.add_attribute("mask", vartype="float", ndim=1, data=mask)
        self.add_varying("vmask", vartype="float", ndim=1)
        
        
        self.add_attribute("cluster", vartype="int", ndim=1, data=cluster)
        self.add_attribute("channel", vartype="int", ndim=1, data=channel)
        
        self.add_attribute("highlight", vartype="int", ndim=1, data=highlight)
        self.add_varying("vhighlight", vartype="int", ndim=1)
        
        
        self.add_uniform("nclusters", vartype="int", ndim=1, data=nclusters)
        # self.add_uniform("nchannels", vartype="int", ndim=1, data=nchannels)
        self.add_uniform("box_size", vartype="float", ndim=2)
        self.add_uniform("box_size_margin", vartype="float", ndim=2)
        self.add_uniform("probe_scale", vartype="float", ndim=2)
        self.add_uniform("superimposed", vartype="bool", ndim=1)
        # self.add_uniform("cluster_colors", vartype="float", ndim=3,
            # size=MAX_CLUSTERS)
        self.add_uniform("channel_positions", vartype="float", ndim=2,
            size=nchannels)
        
        # self.add_varying("varying_color", vartype="float", ndim=4)
        
        
        ncolors = scolors.COLORMAP.shape[0]
        ncomponents = scolors.COLORMAP.shape[1]
        
        
        colormap = scolors.COLORMAP.reshape((1, ncolors, ncomponents))
        
        cmap_index = cluster_colors[cluster]
        
        self.add_texture('cmap', ncomponents=ncomponents, ndim=1, data=colormap)
        self.add_attribute('cmap_index', ndim=1, vartype='int', data=cmap_index)
        self.add_varying('cmap_vindex', vartype='int', ndim=1)
        
        
        dx = 1. / ncolors
        offset = dx / 2.
        
        # print ncolors, dx, offset
        
        global FRAGMENT_SHADER
        FRAGMENT_SHADER = FRAGMENT_SHADER.replace('%CMAP_OFFSET%', "%.5f" % offset)
        FRAGMENT_SHADER = FRAGMENT_SHADER.replace('%CMAP_STEP%', "%.5f" % dx)
        
        # necessary so that the navigation shader code is updated
        self.is_position_3D = True
        
        # add hsv rgb conversion routines
        self.add_fragment_header(FSH)
        
        self.add_vertex_main(VERTEX_SHADER)
        self.add_fragment_main(FRAGMENT_SHADER)

    
class WaveformPaintManager(PlotPaintManager):
    
    def get_uniform_value(self, name):
        if name == "box_size":
            size = self.position_manager.load_box_size()
            if size is None:
                w = h = 0.
            else:
                w, h = size
            return (w, h)
        if name == "box_size_margin":
            size = self.position_manager.load_box_size()
            if size is None:
                w = h = 0.
            else:
                w, h = size
            alpha, beta = self.position_manager.alpha, self.position_manager.beta
            return (w * (1 + 2 * alpha), h * (1 + 2 * beta))
        if name == "probe_scale":
            return self.position_manager.probe_scale
        if name == "superimposed":
            return self.position_manager.superposition == 'Superimposed'
        # if name == "cluster_colors":
            # return self.data_manager.cluster_colors
        if name == "channel_positions":
            pos = self.position_manager.get_channel_positions()
            # print pos
            return pos
    
    def auto_update_uniforms(self, *names):
        dic = dict([(name, self.get_uniform_value(name)) for name in names])
        self.set_data(visual='waveforms', **dic)
        
    def initialize(self):
        # self.set_rendering_options(transparency_blendfunc=('ONE_MINUS_DST_ALPHA', 'ONE'))
        
        self.add_visual(WaveformVisual, name='waveforms',
            npoints=self.data_manager.npoints,
            nchannels=self.data_manager.nchannels,
            nclusters=self.data_manager.nclusters,
            nsamples=self.data_manager.nsamples,
            # nspikes=self.data_manager.nspikes,
            position0=self.data_manager.normalized_data,
            cluster_colors=self.data_manager.cluster_colors,
            mask=self.data_manager.full_masks,
            cluster=self.data_manager.full_clusters,
            channel=self.data_manager.full_channels,
            highlight=self.highlight_manager.highlight_mask)
        
        self.auto_update_uniforms("box_size", "box_size_margin", "probe_scale",
            "superimposed", "channel_positions",)
        
    # @profile
    def update(self):
        
        # self.position_manager.update_arrangement()
        
        
        size, bounds = WaveformVisual.get_size_bounds(self.data_manager.nsamples, self.data_manager.npoints)
        # print bounds
        
        cluster = self.data_manager.full_clusters
        cluster_colors = self.data_manager.cluster_colors
        cmap_index = cluster_colors[cluster]
    
        # self.position_manager.find_box_size()
        
        self.set_data(visual='waveforms', 
            size=size,
            bounds=bounds,
            # nchannels=self.data_manager.nchannels,
            nclusters=self.data_manager.nclusters,
            position0=self.data_manager.normalized_data,
            mask=self.data_manager.full_masks,
            cluster=self.data_manager.full_clusters,
            cmap_index=cmap_index,
            # box_size=self.data_manager.box_size,
            # box_size_margin=self.data_manager.box_size_margin,
            # cluster_colors=self.data_manager.cluster_colors,
            channel=self.data_manager.full_channels,
            highlight=self.highlight_manager.highlight_mask)
        self.auto_update_uniforms('box_size', 'box_size_margin',
            # "probe_scale",
            # "superimposed",
            "channel_positions"
            )
                
        
class WaveformInteractionManager(PlotInteractionManager):
    def select_channel(self, coord, xp, yp):
        # normalized coordinates
        xp, yp = self.get_processor('navigation').get_data_coordinates(xp, yp)
        # find closest channel
        channel, cluster_rel = self.position_manager.find_box(xp, yp)
        # emit the ChannelSelection signal
        ssignals.emit(self.parent, 'ProjectionToChange', coord, channel, -1)
    
    def initialize(self):
        self.register('ToggleSuperposition', self.toggle_superposition)
        self.register('ToggleSpatialArrangement', self.toggle_spatial_arrangement)
        self.register('ChangeBoxScale', self.change_box_scale)
        self.register('ChangeProbeScale', self.change_probe_scale)
        self.register('HighlightSpike', self.highlight_spikes)
        self.register('SelectChannel', self.select_channel_callback)
        self.register(None, self.cancel_highlight)
  
    def toggle_superposition(self, parameter):
        self.position_manager.toggle_superposition()
        
    def toggle_spatial_arrangement(self, parameter):
        self.position_manager.toggle_spatial_arrangement()
        
    def change_box_scale(self, parameter):
        self.position_manager.change_box_scale(*parameter)
    
    def change_probe_scale(self, parameter):
        self.position_manager.change_probe_scale(*parameter)
        
    def highlight_spikes(self, parameter):
        self.highlight_manager.highlight(parameter)
        self.cursor = 'CrossCursor'
    
    def select_channel_callback(self, parameter):
        self.select_channel(*parameter)
        
    def cancel_highlight(self, parameter):
        self.highlight_manager.cancel_highlight()
        
    
class WaveformBindings(SpikyBindings):
    def set_zoombox_keyboard(self):
        """Set zoombox bindings with the keyboard."""
        # Idem but with CTRL + left button mouse 
        self.set('RightClickMove', 'ZoomBox',
                    key_modifier='Alt',
                    param_getter=lambda p: (p["mouse_press_position"][0],
                                            p["mouse_press_position"][1],
                                            p["mouse_position"][0],
                                            p["mouse_position"][1]))
                                            
    def set_arrangement_toggling(self):
        # toggle superposition
        self.set('KeyPress',
                 'ToggleSuperposition',
                 key='O')
                 
        # toggle spatial arrangement
        self.set('KeyPress',
                 'ToggleSpatialArrangement',
                 key='G')

    def set_box_scaling(self):
        # change box scale: CTRL + right mouse
        self.set('RightClickMove',
                 'ChangeBoxScale',
                 key_modifier='Control',
                 param_getter=lambda p: (p["mouse_position_diff"][0]*.2,
                                         p["mouse_position_diff"][1]*.2))

    def set_probe_scaling(self):
        # change probe scale: Shift + left mouse
        self.set('RightClickMove',
                 'ChangeProbeScale',
                 key_modifier='Shift',
                 param_getter=lambda p: (p["mouse_position_diff"][0] * 3,
                                         p["mouse_position_diff"][1] * .5))

    def set_highlight(self):
        # highlight
        self.set('MiddleClickMove',
                 'HighlightSpike',
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],
                                         p["mouse_position"][0],
                                         p["mouse_position"][1]))
        
        self.set('LeftClickMove',
                 'HighlightSpike',
                 key_modifier='Control',
                 param_getter=lambda p: (p["mouse_press_position"][0],
                                         p["mouse_press_position"][1],
                                         p["mouse_position"][0],
                                         p["mouse_position"][1]))
        
    def set_channel_selection(self):
        # CTRL + left click for selecting a channel for coordinate X in feature view
        self.set('LeftClick', 'SelectChannel',
                 key_modifier='Control',
                 param_getter=lambda p: (0, p["mouse_position"][0], p["mouse_position"][1]))
        # CTRL + right click for selecting a channel for coordinate Y in feature view
        self.set('RightClick', 'SelectChannel',
                 key_modifier='Control',
                 param_getter=lambda p: (1, p["mouse_position"][0], p["mouse_position"][1]))
        
    def initialize(self):
        # super(WaveformBindings, self).initialize()
        self.set_arrangement_toggling()
        self.set_box_scaling()
        self.set_probe_scaling()
        self.set_highlight()
        self.set_channel_selection()
    
    
class WaveformView(GalryWidget):
    def initialize(self):
        # self.constrain_navigation = True
        self.constrain_ratio = False
        self.activate3D = True
        self.set_bindings(WaveformBindings)
        self.set_companion_classes(
                data_manager=WaveformDataManager,
                position_manager=WaveformPositionManager,
                interaction_manager=WaveformInteractionManager,
                paint_manager=WaveformPaintManager,
                highlight_manager=WaveformHighlightManager,
                )

    # @profile
    def set_data(self, *args, **kwargs):
        self.data_manager.set_data(*args, **kwargs)
        # update?
        if self.initialized:
            self.paint_manager.update()
            self.updateGL()

        
    # Signals-related methods
    # -----------------------
    def highlight_spikes(self, spikes):
        self.highlight_manager.set_highlighted_spikes(spikes, False)
        self.updateGL()
        

class WaveformWidget(VisualizationWidget):
    def create_view(self, dh):
        self.dh = dh
        self.view = WaveformView(getfocus=False)
        
        # load user preferences
        geometry_preferences = self.restore_geometry()
        if geometry_preferences is None:
            geometry_preferences = {}
        # print geometry_preferences
        self.view.set_data(self.dh.waveforms,
                      clusters=self.dh.clusters,
                      cluster_colors=self.dh.cluster_colors,
                      geometrical_positions=self.dh.probe['positions'],
                      masks=self.dh.masks,
                      **geometry_preferences
                      # user preferences
                      # TODO: store this in the data file, or in the user
                      # preferences
                      # spatial_arrangement=geometry_preferences.get('spatial_arrangement', None),
                      # superposition=geometry_preferencesget('superposition', None),
                      # box_size=geometry_preferencesget('box_size', None),
                      # probe_scale=geometry_preferencesget('probe_scale', None)
                      )
        return self.view
        
    def update_view(self, dh=None):
        if dh is not None:
            self.dh = dh
        self.view.set_data(self.dh.waveforms,
                      clusters=self.dh.clusters,
                      clusters_unique=self.dh.clusters_unique,
                      cluster_colors=self.dh.cluster_colors,
                      geometrical_positions=self.dh.probe['positions'],
                      masks=self.dh.masks
                      )
    
    def initialize_connections(self):
        ssignals.SIGNALS.ProjectionChanged.connect(self.slotProjectionChanged, QtCore.Qt.UniqueConnection)
        ssignals.SIGNALS.ClusterSelectionChanged.connect(self.slotClusterSelectionChanged, QtCore.Qt.UniqueConnection)
        
    def slotClusterSelectionChanged(self, sender, clusters):
        self.update_view()
        
    def slotProjectionChanged(self, sender, coord, channel, feature):
        pass
        
        
    # Save and restore geometry
    # -------------------------
    def save_geometry(self):
        geometry_preferences = {
            'spatial_arrangement': self.view.position_manager.spatial_arrangement,
            'superposition': self.view.position_manager.superposition,
            'box_size': self.view.position_manager.load_box_size(),
            'probe_scale': self.view.position_manager.probe_scale,
        }
        stools.SETTINGS.set("waveformWidget/geometry", geometry_preferences)
        
    def restore_geometry(self):
        """Return a dictionary with the user preferences regarding geometry
        in the WaveformView."""
        return stools.SETTINGS.get("waveformWidget/geometry")
        
        