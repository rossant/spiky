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
[György Buzsaki's laboratory](http://www.buzsakilab.com/).

Gallery
-------

[![Screenshot 1](images/thumbnails/img0.png)](images/img0.png)
[![Screenshot 2](images/thumbnails/img1.png)](images/img1.png)
[![Screenshot 3](images/thumbnails/img2.png)](images/img2.png)

Installation
------------

The software is still in developpement, but you can download an experimental
version here. The first step is to install the dependencies, the second
step is to install the software itself.


### Dependencies
  
The following libraries are required:
  
  * Python 2.7
  * Numpy
  * Pandas >= 0.10
  * Matplotlib >= 1.1.1
  * PyOpenGL >= 3.0.1
  * either PyQt4 or PySide

There are two possibilities to install these dependencies.
  
#### Use an all-in-one Python distribution

All these dependencies are included in the 
[Enthought Python Distribution](http://www.enthought.com/products/epd.php) (EPD),
which is free for academics.


#### Install all packages one by one

This solution can be useful if you already have a Python installation and 
only a few packages are missing.

  * On Windows, [find the packages on this webpage](http://www.lfd.uci.edu/~gohlke/pythonlibs/).
  * On OS X, find them on [Homebrew](http://mxcl.github.com/homebrew/) or [Macports](http://www.macports.org/)
  * On Ubuntu, use `apt-get install` with the appropriate package name, which you can find [here](http://packages.ubuntu.com/).


### Packages

Here are the packages:

  * [TGZ](http://spiky.rossant.net/spiky-0.1.0.dev.tar.gz)
  * [ZIP](http://spiky.rossant.net/spiky-0.1.0.dev.zip)
  * [Windows 64 bits](http://spiky.rossant.net/spiky-0.1.0.dev.win-amd64.exe)

To install Spiky:

  * Make sure you've installed all dependencies.
  * Download one of the packages above.
  * Extract the package and do `python setup.py install`.

### Development version

Use this if you want to be able to update with `git pull` (you need git).

  * Clone the repository:
  
        git clone https://github.com/rossant/spiky.git
  
  * Install Spiky with `pip` so that external packages are automatically
    updated (like `qtools` which contains some Qt-related utility functions):
  
        pip install -r requirements.txt

  
### Test that Spiky is well installed
  
To test that the installation went fine, type in a shell the following command:
    
    $ spiky
    
On Windows, you may need to add `C:\Python27\Scripts` in the PATH variable,
as [described here](http://geekswithblogs.net/renso/archive/2009/10/21/how-to-set-the-windows-path-in-windows-7.aspx).

    