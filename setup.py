import os
from distutils.core import setup

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
        package_data={
            'spiky': ['icons/*.png'],
            
            # INCLUDE GALRY
            'galry': ['cursors/*.png', 'icons/*.png'],
            'galry.visuals': ['fontmaps/*.*'],
            'galry.test': ['autosave/*REF.png'],
            
        },
        
        # scripts=[''],
        url='https://github.com/rossant/spikyy',
        license='LICENSE.md',
        description='Spike sorting graphical interface.',
        long_description=LONG_DESCRIPTION,
        install_requires=[
            "numpy >= 1.6",
            "PyOpenGL >= 3.0",
        ],
    )