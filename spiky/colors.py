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

        
    
def hue(H):
    H = H.reshape((-1, 1))
    R = np.abs(H * 6 - 3) - 1;
    G = 2 - np.abs(H * 6 - 2);
    B = 2 - np.abs(H * 6 - 4);
    return np.clip(np.hstack((R,G,B)), 0, 1)
    
def hsv_to_rgb(HSV):
    a = HSV[:,1].reshape((-1, 1))
    b = HSV[:,2].reshape((-1, 1))
    a = np.tile(a, (1, 3))
    b = np.tile(b, (1, 3))
    return ((hue(HSV[:,0]) - 1) * a + 1) * b
        
        
def hsv_rect(hsv, coords):
    # col = hsv_to_rgb(hsv.reshape((1, -1, 3)))[0,:,:]
    col = hsv_to_rgb(hsv)
    # print col
    col = np.clip(col, 0, 1)
    # print hsv
    # print col
    x0, y0, x1, y1 = coords
    # print x0,y0
    a = 2./len(col)
    c = np.zeros((len(col), 4))
    c[:,0] = np.linspace(x0, x1-a, len(col))
    c[:,1] = y0
    c[:,2] = np.linspace(x0+a, x1, len(col))
    c[:,3] = y1
    rectangles(coordinates=c, color=col)
 
        
if __name__ == "__main__":
    """Show all colors
    """
    
    # from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
    
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
    
    # test_colors()
    
    from galry import *
    
    # a = .06
    # x = [0,.2,.4,.5,.6,.8,1.]
    # y = [0,.2-a,.4+a,.5,.6-a,.8+a,1]
    
    # A = 3.
    # B = 1.5
    
    H = np.linspace(0., 1., 20)#[0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    # S = [1.]*len(H)#[.8, .8, .8, .8, .8, .8, .8, .8, .8, .8, .8]
    # V = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
    i = np.arange(len(H))
    H=H[~((i==5) | (i==10) | (i==12) | (i==19))]
    # H=H[i!=6]
    # H=H[~(i==12)]
    # H=H[i!=13]
    # H=H[i!=18]
    # H=H[~i==19]
    # print len(H)
    H = np.repeat(H, 2)
    n = len(H)
    S = 1*np.ones(n)
    V = 1*np.ones(n)
    V[1::2] = .75
    
    # n = 20
    # t = np.linspace(0., 1., n)
    hsv = np.zeros((n, 3))
    # hsv[:,0] = t#np.interp(t, x, y)
    # hsv[:,0] = B/(1+np.exp(-A*(t-.5)))
    hsv[:,0] = H
    hsv[:,1] = S
    hsv[:,2] = V
    
    figure(constrain_navigation=False)
    
    hsv_rect(hsv, (-1,0,1,1))
    
    hsv[:,1] = 0.5 # white -> color
    # hsv[:,2] = .5 # black -> white
    hsv_rect(hsv, (-1,-1,1,0))
    
    ylim(-1,1)
    
    # col = hsv_to_rgb(hsv.reshape((1, -1, 3)))[0,:,:]
    
    # figure()
    # show()
    
    # col = np.array(COLORS)
    
    # a = 2./len(col)
    # c = np.zeros((len(col), 4))
    # c[:,0] = np.linspace(-1, 1.-a, len(col))
    # c[:,1] = -1
    # c[:,2] = np.linspace(-1+a, 1., len(col))
    # c[:,3] = 1
    
    # hsv = rgb_to_hsv(col.reshape((-1,1,3)))
    # hsv[:,0,-1] = .75
    
    # hues = hsv[:,0,0]
    # asort = hues.argsort()
    # hsv[:,0,0] = hues[asort]
    # hsv[:,0,1] = hsv[asort,0,1]
    
    # rgb = hsv_to_rgb(hsv)[:,0,:]
    
    # figure()
    # rectangles(coordinates=c, color=col)
    
    
    # plot(2*t-1,2*hsv[:,0]-1)
    
    show()
    
    