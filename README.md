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
Inside the CsCl/N4 directory, there's also a jupyter notebook to plot the projected dispersion.

### The config file

To run the projection code, you need the following files:

- eigenvector file: this contains the eigenvectors of the supercell at the Gamma point in CASTEP format. Important: the first part should be cut off: the first line of the file should be the eigenvector for the first mode, first atom.
- eigenvector_directory *(optional)*: optionally you can give a directory containing eigenvector files instead. This is useful in the case where the eigenvectors file is too large to load into memory at once, so here you can split it up into multiple files.
- frequenciesfile: frequencies of phonons of supercell at Gamma point in CASTEP format (this is just the eigenvectors.phonon.csp file, optionally cut after the frequencies to save space.)
- atomlocs_file: .pkl file. The supercell contains N^3*Zp atoms, where Zp is the number of atoms in the primitive unit cell. So for any given atom in Zp, there are N^3 instances of it in the N^3 unit cells in the supercell. This file contains a list: for each atom in Zp, it lists the location in the eigenvectors file in which the atoms of that type occur.
- cellvecs_file: .pkl file. Contains a list: for each of the atoms in Zp, it lists the coordinates of the unit cell corrsponding to the atoms listed in the atomlocs_file. Unit cells are labelled like [0, 0, 0], [1, 0, 0], [0, 1, 0], etc., up to [N-1, N-1, N-1].
- output_directory: path to directory where you want to save projection results.
- kvec_file *(optional)*: if given, the projection will be calculated at these specific (allowed) k-points. Otherwise, it will be calculated at all allowed k-points.

You also need the specify the following settings:
- N: number of unit cells in *each direction* of the supercell. (So total number of unit cells is N^3)
- Z: number of atoms in conventional unit cell
- Z_unitcell: number of atoms in primitive unit cell (=< Z)
- cell_mode: either 'primitive' or 'conventional'. If 'primitive', then all input is assumed to be in primitive cell, and Z_unitcell = Z.
- omegamax: maximum energy value (cm-1) that you want to project
- mode_start: mode at which to start. Normally 1, but could be >1 if you for example want to ignore imaginary modes.
- manual_mode_cutoff *(optional)*: to override omegamax, choose a mode cutoff.

