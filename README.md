# disorderDynamics
Create phonon spectrum for disordered materials.

This code can produce phonon dispersion curves from disordered supercells, projected onto the first Brillouin zone of the unit cell.
Disorder can be of any type: for example mass, force or configurational disorder.
This code is based on the paper by Overy et al.: https://onlinelibrary.wiley.com/doi/abs/10.1002/pssb.201600586.

This code uses supercell eigenvector calculations in CASTEP format. These can be generated with the program GULP: https://gulp.curtin.edu.au/gulp/.

## How to run main code

run the main program code/project.py in a terminal with:

        python project.py <CONFIGPATH> <NSLOTS>

CONFIGPATH is the path to the config file of your system. An example is given for the toy system in toySystems/CsCl/config.ini. It contains paths to all input files and settings to run the projection code. 
To run the toy system, please edit the paths on the config.ini to your local paths. 
NSLOTS is the number of cores for the parallel utility.
