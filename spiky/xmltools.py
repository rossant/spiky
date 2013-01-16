import xml.etree.ElementTree as ET

def parse_xml(filename):

    tree = ET.parse(filename)
    root = tree.getroot()
    
    d = {}

    ac = root.find('acquisitionSystem')
    if ac is not None:
        nc = ac.find('nChannels')
        if nc is not None:
            d['nchannels'] = int(nc.text)
        sr = ac.find('samplingRate')
        if sr is not None:
            d['rate'] = float(sr.text)

    sd = root.find('spikeDetection')
    if sd is not None:
        cg = sd.find('channelGroups')
        if cg is not None:
            g = cg.find('group')
            if g is not None:
                ns = g.find('nSamples')
                if ns is not None:
                    d['nsamples'] = int(ns.text)
                nf = g.find('nFeatures')
                if nf is not None:
                    d['fetdim'] = int(nf.text)
    return d
    
if __name__ == '__main__':
    filename = 'data/test.xml'
    print parse_xml(filename)


