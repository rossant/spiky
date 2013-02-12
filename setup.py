import os
# from distutils.core import setup
from setuptools import *

LONG_DESCRIPTION = """Spike sorting graphical interface."""

if os.path.exists('MANIFEST'):
    os.remove('MANIFEST')

if __name__ == '__main__':

    setup(
        name='spiky',
        version='0.1.0.dev',  # alpha pre-release
        author='Cyrille Rossant',
        author_email='rossant@github',
        packages=['spiky',
                  'spiky.scripts',
                  'spiky.views',
                  
                  # INCLUDE GALRY
                  'galry',
                  'galry.managers',
                  'galry.processors',
                  'galry.python_qt_binding',
                  'galry.test',
                  'galry.visuals',
                  'galry.visuals.fontmaps',
                  
                  ],
        entry_points = {
            'console_scripts': [ 'spiky = spiky.scripts.runspiky:main' ]
        },
        package_data={
            'spiky': ['icons/*.png', '*.css'],
            
            # INCLUDE GALRY
            'galry': ['cursors/*.png', 'icons/*.png'],
            'galry.visuals': ['fontmaps/*.*'],
            'galry.test': ['autosave/*REF.png'],
            
        },
        
        # scripts=['scripts/runspiky.py'],
        
        url='https://github.com/rossant/spikyy',
        license='LICENSE.md',
        description='Spike sorting graphical interface.',
        long_description=LONG_DESCRIPTION,
        install_requires=[
            "numpy >= 1.6",
            "PyOpenGL >= 3.0",
        ],
    )