# import colorsys
import matplotlib.colors as cl
import numpy as np
colconv = cl.ColorConverter()

__all__ = ['COLORMAP', 'COLORS', 'COLORS_COUNT', 'generate_colors']

"""
website: http://phrogz.net/css/distinct-colors.html
"""
COLORS_STRING = """ #ff0000, #0066ff, #00ff22, #ffaa00, #cc00ff, #eeff00, 
#ff0088, #00ffee, #0000d9, #cc2996, #00aaff, #8aa619, #701f99, 
#7f7b40, #406280, #f28100, #e5b800, #d95757, #5283cc, #30bfa3, #996600, 
#7f4040, #488040, #404080, #406280, #0000f2, #8273e6, #cc4514, #cc8166, 
#09b336, #990f46, #7f6240, #40807b, #804073, #139dbf"""    

# generate a list of RGB values for each color
COLORS = map(colconv.to_rgb, map(str.strip, COLORS_STRING.split(",")))
COLORS_COUNT = len(COLORS)

COLORMAP = np.array(COLORS)

def generate_colors(n=None):
    if n is None:
        n = COLORS_COUNT
    if n < COLORS_COUNT:
        return COLORS[:n]
    else:
        return [COLORS[i % COLORS_COUNT] for i in xrange(n)]

if __name__ == "__main__":
    """Show all colors
    """
    def test_colors():
        import matplotlib as mpl
        import pylab as plt
        import numpy as np
        n = len(COLORS)
        
        print "%d colors" % n
        
        x = np.arange(0., n)
        y = np.zeros(n)
        
        x = x.reshape((-1,1))
        y = y.reshape((-1,1))
        
        mpl.rcParams["axes.color_cycle"] = COLORS
        plt.figure(figsize=(14,6))
        axis = plt.subplot(111)
        plt.plot(x.T, y.T, ".", lw=10., ms=10., mew=10.)
        axis.set_axis_bgcolor('k')
        plt.xlim(-1,n)
        plt.show()
    
    test_colors()