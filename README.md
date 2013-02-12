Spiky
=====

*Spiky* is the code name for the next-generation spike-sorting software 
adapted to high-channel count silicon probes. Written in Python, it aims at 
being flexible and extendable by the user.

The first step is to create a new semi-automatic graphical interface
(code name: *KlustaViewa*) for
the manual stage that comes after the automatic clustering algorithms.
This interface automatically guides the user through similar clusters,
showing the most relevant feature projections, and asks him to make merge or
split decisions. The goal is to make the manual stage more reliable, less
error-prone and quicker than what it currently is.

This interface is directly inspired from the current software suite 
[Klusters](http://klusters.sourceforge.net),
[Neuroscope](http://neuroscope.sourceforge.net/)
and
[NDManager](http://ndmanager.sourceforge.net/),
developed in 
[Gy�rgy Buzsaki's laboratory](http://www.buzsakilab.com/).

Gallery
-------

[![Screenshot 1](images/thumbnails/img0.png)](images/img0.png)
[![Screenshot 2](images/thumbnails/img1.png)](images/img1.png)
[![Screenshot 3](images/thumbnails/img2.png)](images/img2.png)

Installation
------------

The software is still in developpement, but you can download an experimental
version here.

### Packages

  * [Windows 64 bits installer](http://spiky.rossant.net/spiky-0.1.0.dev.win-amd64.exe)
  * [ZIP](http://spiky.rossant.net/spiky-0.1.0.dev.tar.gz)
  * [TGZ](http://spiky.rossant.net/spiky-0.1.0.dev.zip)

To launch the software:

    python spiky/scripts/runspiky.py

### Dependencies
  
  * Python 2.7
  * Numpy
  * pandas >= 0.10
  * Matplotlib >= 1.1.1
  * PyOpenGL >= 3.0.1
  * either PyQt4 or PySide
  * h5py
  * galry (included in the package for now)

All dependencies are included in the 
[Enthought Python Distribution](http://www.enthought.com/products/epd.php) (EPD),
which is free for academics.


