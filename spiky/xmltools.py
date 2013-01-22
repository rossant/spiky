import xml.etree.ElementTree as ET

def parse_xml(filename, fileindex=1):

    tree = ET.parse(filename)
    root = tree.getroot()
    
    d = {}

    ac = root.find('acquisitionSystem')
    if ac is not None:
        nc = ac.find('nChannels')
        if nc is not None:
            d['total_channels'] = int(nc.text)
        sr = ac.find('samplingRate')
        if sr is not None:
            d['rate'] = float(sr.text)

    sd = root.find('spikeDetection')
    if sd is not None:
        cg = sd.find('channelGroups')
        if cg is not None:
            # find the group corresponding to the fileindex
            g = cg.findall('group')[fileindex-1]
            if g is not None:
                ns = g.find('nSamples')
                if ns is not None:
                    d['nsamples'] = int(ns.text)
                nf = g.find('nFeatures')
                if nf is not None:
                    d['fetdim'] = int(nf.text)
                c = g.find('channels')
                if c is not None:
                    d['nchannels'] = len(c.findall('channel'))
    return d
    
if __name__ == '__main__':
    filename = 't.xml'
    print parse_xml(filename, 4)


