import numpy as np
import tables
import galry.pyplot as plt

MAXSIZE = 5000

def get_view(total_size, xlim, freq):
    """Return the slice of the data.
    
    Arguments:
      
      * xlim: (x0, x1) of the window currently displayed.
    
    """
    # Viewport.
    x0, x1 = xlim
    d = x1 - x0
    dmax = duration
    zoom = max(dmax / d, 1)
    view_size = total_size / zoom
    step = int(np.ceil(view_size / MAXSIZE))
    # Extended viewport for data.
    x0ex = np.clip(x0 - 2 * d, 0, dmax)
    x1ex = np.clip(x1 + 2 * d, 0, dmax)
    i0 = np.clip(int(np.round(x0ex * freq)), 0, total_size)
    i1 = np.clip(int(np.round(x1ex * freq)), 0, total_size)
    return (x0ex, x1ex), slice(i0, i1, step)

def get_undersampled_data(data, xlim, slice):
    """
    Arguments:
    
      * data: a HDF5 dataset of size Nsamples x Nchannels.
      * xlim: (x0, x1) of the current data view.
      
    """
    # total_size = data.shape[0]
    # Get the view slice.
    # x0ex, x1ex = xlim
    # x0d, x1d = x0ex / (duration_initial) * 2 - 1, x1ex / (duration_initial) * 2 - 1
    # Extract the samples from the data (HDD access).
    samples = data[slice, :]
    # Convert the data into floating points.
    samples = np.array(samples, dtype=np.float32)
    # Normalize the data.
    samples *= (1. / 65535)
    samples *= .25
    # Size of the slice.
    nsamples, nchannels = samples.shape
    # Create the data array for the plot visual.
    M = np.empty((nsamples * nchannels, 2))
    samples = samples.T + np.linspace(-1., 1., nchannels).reshape((-1, 1))
    M[:, 1] = samples.ravel()
    # Generate the x coordinates.
    x = np.arange(slice.start, slice.stop, slice.step) / float(total_size - 1)
    # [0, 1] -> [-1, 2*duration.duration_initial - 1]
    x = x * 2 * duration / duration_initial - 1
    M[:, 0] = np.tile(x, nchannels)
    # Update the bounds.
    bounds = np.arange(nchannels + 1) * nsamples
    size = bounds[-1]
    return M, bounds, size


filename = "test_data/n6mab031109.h5"
f = tables.openFile(filename)
data = f.root.raw_data
nsamples, nchannels = data.shape
total_size = nsamples
freq = 20000.
dt = 1. / freq
duration = (data.shape[0] - 1) * dt

duration_initial = 10.

x = np.tile(np.linspace(0., duration, nsamples // MAXSIZE), (nchannels, 1))
y = np.zeros_like(x)+ np.linspace(-1, 1, nchannels).reshape((-1, 1))

plt.figure(toolbar=False, show_grid=True)
plt.plot(x=x, y=y)

SLICE = None

def anim(figure, parameter):
    # Constrain the zoom.
    nav = figure.get_processor('navigation')
    nav.constrain_navigation = True
    nav.xmin = -1
    nav.xmax = 2 * duration / duration_initial
    nav.sxmin = 1.
    
    zoom = nav.sx
    box = nav.get_viewbox()
    xlim = ((box[0] + 1) / 2. * (duration_initial), (box[2] + 1) / 2. * (duration_initial))
    xlimex, slice = get_view(data.shape[0], xlim, freq)
    global SLICE
    if slice != SLICE:
        SLICE = slice
        samples, bounds, size = get_undersampled_data(data, xlimex, slice)
        nsamples = samples.shape[0]
        color_array_index = np.repeat(np.arange(nchannels), nsamples / nchannels)
        figure.set_data(position=samples, bounds=bounds, size=size,
            index=color_array_index)
    
plt.animate(anim, dt=.25)

plt.xlim(0., 10.)

plt.show()
f.close()
