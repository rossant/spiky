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
    total_size = data.shape[0]
    # Get the view slice.
    x0ex, x1ex = xlim
    x0d, x1d = x0ex / (duration) * 2 - 1, x1ex / (duration) * 2 - 1
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
    x = np.arange(slice.start, slice.stop, slice.step) / float(total_size - 1) * 2 - 1
    M[:, 0] = np.tile(x, nchannels)
    # Update the bounds.
    bounds = np.arange(nchannels + 1) * nsamples
    size = bounds[-1]
    return M, bounds, size


filename = "test_data/n6mab031109.trim.h5"
f = tables.openFile(filename)
data = f.root.RawData
nsamples, nchannels = data.shape
freq = 20000.
dt = 1. / freq
duration = (data.shape[0] - 1) * dt

# Convert the data into floating points.
step = nsamples // MAXSIZE
samples = np.array(data[::step, :], dtype=np.float32).T
# Normalize the data.
samples *= (1. / 65535)
samples *= .25

# |_|_|_|
# n = 4
# duration = (n-1)*dt
# xmax = duration

y = samples + np.linspace(-1, 1, nchannels).reshape((-1, 1))
x = np.tile(np.linspace(0., duration, y.shape[1]), (nchannels, 1))

plt.plot(x=x, y=y)

SLICE = None

def anim(figure, parameter):
    zoom = figure.get_processor('navigation').sx
    box = figure.get_processor('navigation').get_viewbox()
    xlim = ((box[0] + 1) / 2. * (duration), (box[2] + 1) / 2. * (duration))
    xlimex, slice = get_view(data.shape[0], xlim, freq)
    global SLICE
    if slice != SLICE:
        SLICE = slice
        samples, bounds, size = get_undersampled_data(data, xlimex, slice)
        nsamples = samples.shape[0]
        color_array_index = np.repeat(np.arange(nchannels), nsamples / nchannels)
        figure.set_data(position=samples, bounds=bounds, size=size,
            index=color_array_index)
    
plt.animate(anim, dt=1.)

plt.show()
f.close()
