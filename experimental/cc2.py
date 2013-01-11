
from pylab import *
import numpy as np
from numba import *
from numpy import random

def cc(T1, T2, width=.02, bin=.001, T=None):
    n = int(np.ceil(width / bin)) # Histogram length
    if (len(T1) == 0) or (len(T2) == 0): # empty spike train
        return None
    i = 0
    j = 0
    l = []
    for t in T1:
        while i < len(T2) and T2[i] < t - width: # other possibility use searchsorted
            i += 1
        while j < len(T2) and T2[j] < t + width:
            j += 1
        l.extend(T2[i:j] - t)
    H, _ = np.histogram(l, bins=np.arange(2 * n + 1) * bin - n * bin) #, new = True)
    # H = zeros(2*n)
    bins = np.arange(2 * n + 1) * bin - n * bin
    for i in xrange(len(bins)-1):
        t0, t1 = bins[i], bins[i+1]
    return H#*1./H.max()

@jit(double[:](double[:], double[:]))
def cc2(T1, T2):
    width=.02
    bin=.001
    
    n = int(ceil(width / bin)) # Histogram length
    
    i = 0
    j = 0
    n2 = 2 * n + 1
    
    bins = arange(n2) * bin - n * bin
    H = zeros(len(bins)-1)
    
    for a in range(len(T1)):
        t = T1[a]
        while i < len(T2) and T2[i] < t - width: # other possibility use searchsorted
            i += 1
        while j < len(T2) and T2[j] < t + width:
            j += 1
        for h in range(j-i):
            x = T2[i+h] - t
            # u = max(0, min(int(floor(x/bin))+n, 2*n-1))
            u = max(0, min(x/bin+n, n2))
            H[u] += 1
    
    return H#/H.max()
    
# random.seed(11012013)
t1=cumsum(random.exponential(scale=.01, size=10000))
t2=cumsum(random.exponential(scale=.01, size=10000))
t2 = hstack((t2, t1[::10]))
t2.sort()

import time
tt0 = time.clock()
C=cc(t1, t2)
tt1 = time.clock()
C2=cc2(t1, t2)
tt2 = time.clock()

print "brian", tt1-tt0
print "numba", tt2-tt1
print abs(C-C2).max()

# plot(C)
# plot(C2)
# show()


