Spiky: graphical interface for semi-automatic spike sorting
===========================================================

TODO
----  


  * no firing rate but n of spikes in clusterview
  * CTRL + A to select all spikes + button
  * feature view: normalize everything once at initialization time
  * normalize correlograms
  * feature projection box: TAB goes from X to Y channel selection
  * open klusters file

  * fix focus issue with floating docks
  
  * option to toggle showing masks as gray waveforms in waveform view
  * option to select the max number of waveforms to display (selected at
    random)
  
  * selection of cluster: highlight also in waveformview
  * optional color for groups, and option to use that color for all clusters
    in that group on the feature view. ON by default for MUA.
  
  * default groups: good, noise (SH DEL), multiunit (DEL) move to groups
  * possibility to change the color of clusters permanently
  
  * interaction mode buttons: deactivate in gray to highlight the current mode
  * buttons for all commands (reset view for any view)
  * highlighting: less transient with CTRL + click (click to deactivate)
  * cluster view: left arrow for closing the tree view
  
  * display mean/std waveform (see klusters)
  * option to change width/bin of correlograms

  * function for automatic zoom in waveformview as a function of
    channels and clusters
  * cluster group renaming
  * featureview controller: buttons for channel up/down
  * small widget with text information (number of spikes, highlighted spikes...)
  * cluster merge with button / M press
  * cluster split with button / S press
  * split with different clusters: split everything
  
  
File format
-----------

  * use the existing XML file (reverse engineer) and store all information
    related to visualization in here (cluster colors, probe scaling, etc)
  * keep the old clu file and create a new one
  
  
Ideas
-----
  
  * Measure of cluster quality: ratio of mask/unmask on each channel
  * correlation matrix: later (lower priority)
  * feature view: when masks toggled (features gray) not possible to select
    them. when no masks, everything can be selected.
  * trace view (neuroscope)
  * fetdim variable in controller widget (1 to ?)


Fixes
-----

  * make sure the GUI work in IPython
  * force cleaning up of widgets upon closing

  
Refactoring
-----------

  * refactor interactions in waveformview/featureview with different
    processors...
  * refactoring correlograms: put the actual computations outside dataio
  * put COLORMAP in the data holder and remove dependencies 
  * factorize highlight logic in shaders  
  * move data manager into templates, so that templates contain everything


Multiplatform notes
-------------------

  * pyside: crash when running the software = erase the .INI file
  * PySide: bug when runnin python test.py, one needs to do ipython then run...
  * macosx/nvidia: galry 3D examples look funny
  * INI file bug between pyqt and pyside related to waveformview geometry saving
    make sure to use native formats for objects
  * windows/pyside cluster view: cluster selection not blue but gray (style not working?)
  * initial launch: default window geometry config

  