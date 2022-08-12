# PROJECT PHONON MODES OF SUPERCELL ONTO FIRST BZ OF UNIT CELL
# HOW TO RUN: python project.py <CONFIGPATH> <NSLOTS>
# CONFIGPATH = path to config.ini file of your system
# NSLOTS = number of cores for parallel functionality

import pandas as pd
import numpy as np
import sys
import os
import time
import h5py
import pickle
from numba import jit, prange
from joblib import Parallel, delayed
sys.path.append('./')
import kpoints_fcc
import configparser

configpath = sys.argv[1]
nslots = int(sys.argv[2])


""" read input from config file """

config = configparser.ConfigParser()
config.read(configpath)

# input files
# can input an eigenvector file or an eigenvector directory
try:
    pfile = config['files']['eigenvectorfile']
    filemethod = 'fromfile'
except:
    eigfile_dir = config['files']['eigenvector_directory']
    filemethod = 'fromdirectory'
freq_file = config['files']['frequenciesfile']
atomlocs_file = config['files']['atomlocs_file']
cellvecs_file = config['files']['cellvecs_file']
outdir = config['files']['output_directory']

# settings
N = int(config['settings']['N'])
Z = int(config['settings']['Z'])
Zp = int(config['settings']['Z_unitcell'])
MODE = config['settings']['cell_mode']  # primitive or conventional
omegamax = float(config['settings']['omegamax'])
mode_start = int(config['settings']['mode_start'])
try:
    manual_mode_cutoff = int(config['settings']['manual_mode_cutoff'])  # up to and including this mode
except:
    print('no manual mode cutoff found - using omega max')

# kpoints
# can optionally give a kvec file with specific k-points. Otherwise calculated all allowed k-points
try:
    k_file = config['files']['kvec_file']
    with open(k_file, 'rb') as handle:
        k_fcc = pickle.load(handle)
except:
    print('no kvec file found - calculating all allowed k-points')
    k_fcc = np.array(kpoints_fcc.all_k_in_zone(N))
print('cell mode is {}'.format(MODE))
if MODE=='primitive':
    M = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])  # reciprocal FCC - P1 transformation matrix
    k_p = []
    for k in k_fcc:
        k_p.append(np.matmul(M, k))
    k_vecs = np.vstack(k_p)
elif MODE=='conventional':
    k_vecs = k_fcc


""" functions """


def cellAmplitude(eig, k, r):
    """ Calculate the function eig* exp(i*k*r) for a certain cell l.
    Parameters
    ----------
    eig: float
        entry of the relevant eigenvector
    k: array (size=3)
        k-vector
    r: array (size=3)
        position vector of relevant cell
    """
    val = eig * np.exp(1j * np.dot(k, r))
    return val


@jit(nopython=True, parallel=True)  # Set "nopython" mode for best performance, equivalent to @njit
def make_exponential(N, cell_vecs, k_vecs):
    """Collect all k-vectors and r-vectors, and do the exponential.
    Parameters
    ----------
    N: int
        size of NxNxN supercell
    r_vecs: numpy array
        size (n_atoms, 3): fractional coordinates of all atoms in system
        can be read from coordinates.pkl file created in get_locations()

    Returns
    -------
    exp_array: np.array
        array of exponentials, where the first index is k-vector,
        second index is the atom id jl
    """
    # here we call the k points from the kpoints file
    # cell_vecs = []
    # for x in range(N):
    #     for y in range(N):
    #         for z in range(N):
    #             k_vecs.append([x, y, z])
    # cell_vecs = np.array(cell_vecs)

    exponentialMatrix = np.exp(1j * 2 * np.pi * np.dot(k_vecs, cell_vecs.T))
    # now we have an array where first index is k, second index is r

    return exponentialMatrix, k_vecs


def read_and_project_mode(mode, N, Z, exp_array, kvecs, type_indices, method='fromfile'):
    """Read the phonon output file and for the given mode, reorganise it into
    a big dataframe with entries mode, omega, alpha, j, l, eigenvector entry
    To do: get j and l from ion number.
    j = which atom in cell
    l = which cell"""

    print('calculating mode {}'.format(mode))

    # select the correct file
    if method == 'fromfile':
        ef = open(pfile, 'r')
        file_startmode=1
        this_eigfile = pfile
    else:
        for eigfile in os.listdir(eigfile_dir):
            if 'frequencies' in eigfile:
                continue
            numbers = eigfile.replace('eigslice_mode', '').replace('.csp', '').split('-')
            n1 = int(numbers[0])
            n2 = int(numbers[1])
            if n1 <= mode <= n2:
                file_startmode = n1
                file_endmode = n2
                this_eigfile = os.path.join(eigfile_dir, eigfile)
                break
        ef = open(this_eigfile, 'r')

    # get target lines for the case where you have multiple sub-eigfiles
    target_atomlines = []
    for atom in range(1, N**3*Z+1):
        target_atomline = (mode-file_startmode) * N ** 3 * Z + atom - 1  # -1 because lines start at 0
        # print line for verification
        if mode == 1 and atom == 1:
            print('target atomline mode 1, atom 1 in file {} is {}'.format(this_eigfile, target_atomline))
        target_atomlines.append(target_atomline)
    target_energyline = 13 + N ** 3 * Z + 1 + mode - 1  # -1 because lines start at 0

    all_ex = []
    all_ey = []
    all_ez = []

    # read frequency from frequencies file
    with open(freq_file, 'r') as ff:
        for i, line in enumerate(ff):
            if i == target_energyline:
                terms = line.split()
                omega = float(terms[1])
                if omega > omegamax:
                    print('reached omegamax = {} at mode {}'.format(omegamax, mode))
                    return None, None, None, True

    # get data from eigenvectors file
    for i, line in enumerate(ef):

        if i > target_atomlines[-1]:
            # then we've read all relevant lines
            break

        elif target_atomlines[0] <= i <= target_atomlines[-1]:

            terms = line.split()

            # now check how many terms: if numbers are too large (happens when there is too many atoms in system),
            # then first two terms are merged.
            # normal case: 8 terms. merged case: 7 terms
            if len(terms) == 8:
                ex = float(terms[2]) + 1j * float(terms[3])
                ey = float(terms[4]) + 1j * float(terms[5])
                ez = float(terms[6]) + 1j * float(terms[7])
            elif len(terms) == 7:
                ex = float(terms[1]) + 1j * float(terms[2])
                ey = float(terms[3]) + 1j * float(terms[4])
                ez = float(terms[5]) + 1j * float(terms[6])

            all_ex.append(ex)
            all_ey.append(ey)
            all_ez.append(ez)

    mode_result = calc_proj(np.array(all_ex), np.array(all_ey), np.array(all_ez), exp_array, len(kvecs), type_indices)

    ef.close()
    return mode, mode_result, omega, False


def calc_proj(all_ex, all_ey, all_ez, exp_array, n_kvecs, type_indices):
    # calculate projections
    mode_result = np.zeros(n_kvecs)  # length of number of k-vectors
    # sum over alpha
    for eigenvectors in [all_ex, all_ey, all_ez]:
        # multiply by exponential and sum

        # now sum over the the unit cell atoms
        for atom in range(Zp):
            
            # get eigenvectors of this atom type
            myindices = np.array(type_indices[atom]) ## - 1 # the new_indices start at 0!! so no need for '-1'
            eigs = eigenvectors[myindices]
            
            # this dot product includes our sum over cells
            amplitude = np.dot(exp_array[atom], eigs.T)  # should result in array of size (len(kvecs))
                
            mode_result += (np.abs(amplitude)**2).real

    mode_result = mode_result / (N ** 3)  # divide by number of unit cells in supercell
    return mode_result


def calculate_max_mode(phononfile, N, Z, omegamax):
    """Calculate the maximum mode number that we need to project.
    phononfile: string
        path to eigenvectors file
    N: int
        supercell size
    Z: int
        number of atoms in unit cell
    omegamax: float
        maximum energy value (in castep cm-1 units)
    """

    # mode 1 is at (13 + N ** 3 * Z + 1 + mode - 1) = (13 + N ** 3 * Z + 1)
    first_mode = 13 + N ** 3 * Z + 1
    last_mode = 3*N**3*Z
    last_mode_loc = 13 + N**3*Z + last_mode
    with open(phononfile, 'r') as ef:
        for i, line in enumerate(ef):

            if i >= first_mode:
                if i > last_mode_loc:
                    print('including all modes')
                    return last_mode
                terms = line.split()
                mode = int(terms[0])
                omega = float(terms[1])
                if omega > omegamax:
                    print('reached omegamax = {} at mode {}'.format(omegamax, mode))
                    return mode


def extend_projections(kvecs, all_modes, all_projections, all_omegas):
    # extend kvecs
    # all_kvecs = np.tile(kvecs, (len(all_omegas), 1))  # np.tile not supported in numba. need to do a workaround:
    tiled = np.zeros((len(all_omegas), kvecs.shape[0], kvecs.shape[1]))
    for dim in range(len(all_omegas)):
        tiled[dim, :, :] = kvecs
    all_kvecs = tiled.reshape((len(all_omegas) * len(kvecs), 3))
    # and unfold projections and omegas into list of items
    projections_big = all_projections.reshape((len(all_omegas) * len(kvecs), ))
    omegas_big = np.repeat(all_omegas, len(kvecs))
    modes_big = np.repeat(all_modes, len(kvecs))
    return modes_big, projections_big, all_kvecs, omegas_big


def save_projections(kvecs, all_modes, all_projections, all_omegas, outdir):
    # save them into different files: modes, projections, kvecs, omegas
    # could also make it different datasets in h5, might be faster?
    names = ['modes', 'projections', 'kvecs', 'omegas']
    objects = extend_projections(kvecs, np.array(all_modes), np.array(all_projections), np.array(all_omegas))
    for ii, obj in enumerate(objects):
        print(names[ii], obj)
        with h5py.File(os.path.join(outdir, '{}.h5'.format(names[ii])), 'w') as hf:
            hf.create_dataset("{}".format(names[ii]), data=obj)


def project(k_vecs):

    # get cell coordinates
    with open(cellvecs_file, 'rb') as handle:
        cell_vecs = pickle.load(handle)
    # expected shape: (Zp, N**3, 3)
    if np.array(cell_vecs).shape == tuple([N**3, 3]):
        cell_vecs = np.tile(np.array(cell_vecs), (Zp, 1, 1))

    # get exp_array
    exp_array = []
    for vecs in cell_vecs:
        exp, kvecs = make_exponential(N, vecs, k_vecs)
        exp_array.append(exp)
    exp_array = np.array(exp_array)

    # calculate maximum mode we need to project
    max_mode = calculate_max_mode(pfile, N, Z, omegamax)

    # get type indices
    with open(atomlocs_file, 'rb') as handle:
        type_indices = pickle.load(handle)

    # parallel loop over modes
    try:
        max_mode = manual_mode_cutoff + 1
    except Exception:
        pass
    results = Parallel(n_jobs=nslots)(delayed(read_and_project_mode)(m, N, Z, exp_array, kvecs, type_indices, method=filemethod) for m in range(mode_start, max_mode))
    all_modes = [x[0] for x in results]
    all_projections = [x[1] for x in results]
    all_omegas = [x[2] for x in results]
    save_projections(kvecs, all_modes, all_projections, all_omegas, outdir)
    return


project(k_vecs)
