import collections
import numpy as np
from matplotlib.pyplot import imread
from galry import *
from spiky import *


def generate_colors(n=None):
    if n is None:
        n = COLORS_COUNT
    if n < COLORS_COUNT:
        return COLORMAP[:n]
    else:
        return [COLORMAP[i % COLORS_COUNT] for i in xrange(n)]

"""

Toolbar:

  * dropdown: select shank, used for "new channel" button
  * button: load/save
  
"""

def get_disc(size):
    texture = np.zeros((size, size, 4))
    
    # force even number
    size -= (np.mod(size, 2))
    # fill with white
    texture[:, :, :-1] = 1
    x = np.linspace(-1., 1., size)
    X, Y = np.meshgrid(x, x)
    R = X ** 2 + Y ** 2
    R = np.minimum(1, 20 * np.exp(-8*R))
    # disc-shaped alpha channel
    texture[:size,:size,-1] = R
    return texture

    
class Probe(Manager):
    
    def load(self, imgname=None):
        self.image = imread(imgname)

    def save(self, filename):
        np.savetxt(filename, self.get_positions(), fmt='%.3f')
        
    def add_channel(self, position=(0, 0), shank=0):
        self.positions = np.vstack((self.get_positions(), position))
        self.shanks = np.hstack((self.get_shanks(), shank))
        
    def remove_channel(self, channelidx):
        indices = range(self.get_nchannels())
        del indices[channelidx]
        if len(indices) > 0:
            self.positions = np.vstack(self.get_positions()[indices,:])
            self.shanks = np.hstack(self.get_shanks()[indices])
        
    def move_channel(self, channelidx, position):
        self.positions[channelidx, :] = position
        
    def get_image(self):
        return self.image
        
    def get_positions(self):
        if not hasattr(self, 'positions'):
            return np.zeros((0, 2))
        return self.positions

    def get_shanks(self):
        if not hasattr(self, 'shanks'):
            return np.zeros(0, dtype=np.int32)
        return self.shanks

    def get_nshanks(self):
        shanks = self.get_shanks()
        counter = collections.Counter(shanks)
        nshanks = max(counter.keys()) + 1
        return nshanks
        
    def get_text(self):
        return map(str, range(1, self.get_nchannels() + 1))
        
    def get_colors(self):
        nshanks = self.get_nshanks()
        colors = generate_colors(nshanks)
        colors = np.hstack((colors, .75 * np.ones((nshanks, 1))))
        return colors[self.shanks, :]
        
    def get_nchannels(self):
        return self.get_positions().shape[0]
        
        
class ProbePM(PaintManager):
    def initialize(self):
        # probe = Probe()
        probe = self.probe_manager
        
        # MOCK
        probe.load('probe.png')
        probe.add_channel()
        text = probe.get_text()
        colors = probe.get_colors()
        # print text
        # print colors.shape, colors.dtype
        # print list(colors)
        
        self.add_visual(TextureVisual, texture=probe.get_image(), is_static=True,
                mipmap=True,
                minfilter='LINEAR_MIPMAP_NEAREST',
                magfilter='LINEAR', name='probe_image')
        self.add_visual(SpriteVisual, position=probe.get_positions(), 
            texture=get_disc(50), color=colors, name='channels')
        self.add_visual(TextVisual, coordinates=probe.get_positions(), 
            text=text, posoffset=(.01, .005),
            fontsize=18, letter_spacing=300.,
            color=get_color('w'), 
            name='channels_text')
        w, h = .2, .08
        self.add_visual(RectanglesVisual,
            coordinates=(-.8-w, .9-h, -.8+w, .9+h), color='k')
        self.add_visual(TextVisual, coordinates=(-.8, .9), 
            text='Shank 0', 
            fontsize=24, #letter_spacing=300.,
            color=get_color('y'), 
            name='shank_text')
    
    
class ProbeIM(InteractionManager):
    def initialize(self):
        self.selected_channel = None
        self.shank = 0
        self.nshanks = 1
        self.register(None, self.cancel)
        self.register('MoveChannel', self.move_channel)
        self.register('AddChannel', self.add_channel)
        self.register('RemoveChannel', self.remove_channel)
        self.register('AddShank', self.add_shank)
        self.register('NextShank', self.next_shank)

    def move_channel(self, position):
        probe = self.probe_manager
        positions = probe.get_positions()
        
        if positions.size > 0:
            # select channel
            if self.selected_channel is None:
                d = np.sum((positions - position) ** 2, axis=1)
                channelidx = np.argmin(d)
                if d[channelidx] > .01:
                    return
                self.selected_channel = channelidx
            # or use already selected channel
            else:
                channelidx = self.selected_channel
            
            # update the positions
            self.probe_manager.move_channel(channelidx, position)
            # update the probe
            self.update_probe()
        
    def add_channel(self, position):
        probe = self.probe_manager
        probe.add_channel(position, self.shank)
        self.update_probe()
    
    def remove_channel(self, position):
        probe = self.probe_manager
        positions = probe.get_positions()
        if positions.size > 0:
            d = np.sum((positions - position) ** 2, axis=1)
            channelidx = np.argmin(d)
            if d[channelidx] > .01:
                return
            probe.remove_channel(channelidx)
            self.update_probe()
    
    def cancel(self, param=None):
        self.selected_channel = None
        
    def update_probe(self):
        probe = self.probe_manager
        positions = probe.get_positions()
        colors = probe.get_colors()
        text = probe.get_text()
        self.paint_manager.set_data(position=positions,
            color=colors,
            visual='channels')
        self.paint_manager.set_data(coordinates=positions, 
            text=text,
            visual='channels_text')
        
    def update_shank(self):
        probe = self.probe_manager
        nshanks = probe.get_nshanks()
        # color = generate_colors(nshanks)[self.shank - 1,:]
        # print color
        self.paint_manager.set_data(text='Shank {0:d}'.format(self.shank),
            visual='shank_text', 
            # color=color
            )
        
    def add_shank(self, param):
        self.shank += 1
        self.nshanks += 1
        self.update_shank()
        
    def next_shank(self, param):
        self.shank = np.mod(self.shank + param, self.nshanks)
        self.update_shank()
        

class ProbeBindings(Bindings):
    def initialize(self):
        self.set('LeftClickMove', 'MoveChannel',
            param_getter=lambda p: p['mouse_position'])
        self.set('LeftClick', 'AddChannel',
            param_getter=lambda p: p['mouse_position'])
        self.set('RightClick', 'RemoveChannel',
            param_getter=lambda p: p['mouse_position'])
        self.set('KeyPress', 'AddShank', key='S')
        self.set('KeyPress', 'NextShank', key='Left', param_getter=-1)
        self.set('KeyPress', 'NextShank', key='Right', param_getter=1)
    
    
class ProbeWidget(GalryWidget):
    def initialize(self):
        self.constrain_ratio = True
        self.toolbar = None
        self.set_companion_classes(
            paint_manager=ProbePM,
            interaction_manager=ProbeIM,
            probe_manager=Probe,
            )
        self.set_bindings(ProbeBindings)
        

        
show_basic_window(ProbeWidget)

