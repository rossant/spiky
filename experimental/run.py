import sys
import re
import numpy as np
from spiky import *

# file = r"D:\Spike sorting\n6mab031109_buzsaki32\n6mab031109_buzsaki32.h5"


        
        
        
if __name__ == '__main__':
    
    filename = ""
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
        
    r = re.search(r"([^\n]+)\.[^\.]+\.[0-9]+$", filename)
    if r:
        filename = r.group(1)
        
    if filename:
        class MySpiky(SpikyMainWindow):
            def initialize_data(self):
                provider = KlustersDataProvider()
                self.dh = provider.load(filename)
                self.sdh = SelectDataHolder(self.dh)
    else:
        MySpiky = SpikyMainWindow
        
    window = show_window(MySpiky)
    