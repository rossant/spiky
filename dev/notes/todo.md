Spiky: graphical interface for semi-automatic spike sorting
===========================================================

Message for alpha pre-release version:

    This is an alpha version of a software that is still in development.
    Some features are missing, bugs and crashes may appear, and the software
    may be slow with large data sets. 
    These issues will be fixed in later releases.
    Make sure you backup your data before loading it (although the software
    won't modify any of your files by default).
    
Before alpha pre-release
------------------------

  * invalidate correlograms when merging/splitting clusters
  * load probe file

  * correlogram computations in external thread, signal in thread to signal
    that the correlograms view needs to be updated
  * test with big data sets

  * initial launch: default window geometry config
  
  
Major features
--------------

  * GUI to visualize probe files
  * new HDF5 file format
  * robot

  
Minor features
--------------
  
  * selection of cluster: highlight also in waveformview
  * use the existing XML file (reverse engineer) and store all information
    related to visualization in here (cluster colors, probe scaling, etc)
  
  * feature view and waveform view: display cluster index of the closest object
    from the mouse, when pressing a button
  * buttons for all commands (reset view for any view)
  * interaction mode buttons: deactivate in gray to highlight the current mode
  
  * display mean/std waveform (see klusters)
  * highlighting: less transient with CTRL + click (click to deactivate)
  * option to change width/bin of correlograms
  * option to toggle showing masks as gray waveforms in waveform view
  * option to select the max number of waveforms to display (selected at
    random)
  * function for automatic zoom in waveformview as a function of
    channels and clusters
  * small widget with text information (number of spikes, highlighted spikes...)


Improvements
------------

  * feature view gray points transparency depth
  * correlogram computations in external thread

  
Optimizations
-------------

  * improve the computation of cross-correlograms (specific algorithm for
    pairwise correlograms within a pool of neurons, maybe with numba)


Fixes
-----

  * make sure the GUI work in IPython
  * force cleaning up of widgets upon closing
  * fix focus issue with floating docks
  
  
Refactoring
-----------

  * in gui.py, put all actions in a separate class, and avoid communicating 
    directly from mainwindow to widgets, rather, use signals
  * refactor interactions in waveformview/featureview with different
    processors...
  * refactoring correlograms: put the actual computations outside dataio
  * put COLORMAP in the data holder and remove dependencies 
  * move data manager into templates, so that templates contain everything


Multiplatform notes
-------------------

  * pyside: crash when running the software = erase the .INI file
  * PySide: bug when runnin python test.py, one needs to do ipython then run...
  * windows/pyside cluster view: cluster selection not blue but gray (style not working?)

  
Ideas
-----
  
  * Measure of cluster quality: ratio of mask/unmask on each channel
  * correlation matrix: later (lower priority)
  * feature view: when masks toggled (features gray) not possible to select
    them. when no masks, everything can be selected.
  * trace view (neuroscope)
  * fetdim variable in controller widget (1 to ?)

