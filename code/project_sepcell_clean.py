# python script to calculate projections, given that the eigenvector file has been 
# split up by mode ranges
# IMPORTANT: every one of those file should start precisely with the first eigenvector of the first mode, otherwise this code will select the wrong line numbers

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
#import kpoints


# TO DO: add projection code here

nslots = int(sys.argv[1])
# pfile = str(sys.argv[2])
# N = int(sys.argv[3])
# Z = int(sys.argv[4])
# outdir = str(sys.argv[5])
# omegamax = float(sys.argv[6])*8.065  # convert from meV to cm-1

N = 4
Z = 2
wd = './N{}'.format(N)
pfile = os.path.join(wd, 'only_eigenvectors.csp')
eigfile_dir = '/data/scratch/apw690/adagulp_HT/phonon_output/production/N8_0_cores216/sliced_eigfiles/'
freq_file = os.path.join(wd, 'frequencies.csp')
labels_file = os.path.join(wd, 'labels.txt')
outdir = os.path.join(wd, 'projection')
omegamax = 150*8.065  # convert from meV to cm-1
mode_start = 1
# cutoffs: N8: 3072, N4: 384, N1: 6
manual_mode_cutoff = 384  # including this mode!!
#manual_mode_cutoff = 580
#k_vecs = np.array(kpoints.all_k_in_zone(N))[0:3]
#k_vecs = np.array([[0.0, 0.0, 0.0], [0.125, 0.0, 0.0], [0.25, 0.0, 0.0], [0.375, 0.0, 0.0], [0.5, 0.0, 0.0]])
k_vecs = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0], [0.75, 0.0, 0.0]])
#k_vecs = np.array([[0.0, 0.0, 0.0]])


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


def make_indices(N, Z):
    all_idxs = []
    for atom in range(Z):
        idxs = [atom + Z*cc for cc in range(N)]
        all_idxs.append(idxs)
    return all_idxs


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


#@profile
def get_labels(labelsfile):
    with open(labelsfile, 'rb') as f:
        df = pd.read_pickle(f)  # columns=['atom_id', 'atom', 'molecule', 'cell', 'orientation'])
    return df


def read_and_project_mode(mode, N, Z, exp_array, kvecs, type_indices, method='file'):
    """Read the phonon output file and for the given mode, reorganise it into
    a big dataframe with entries mode, omega, alpha, j, l, eigenvector entry
    To do: get j and l from ion number.
    j = which atom in cell
    l = which cell"""

    print('calculating mode {}'.format(mode))

    # select the correct file
    if method=='file':
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
            # i don't think I even need the atom idx. I just need to know if the value
            # is in the list. I might delete value from list after it,
            # which will make the next search quicker?
            # or actually, I could just do a statement lower < i < upper
            #try:
            #    atom_idx = target_atomlines.index(i) + 1
            #except ValueError:
            #    continue

            terms = line.split()

            # now check how many terms: if numbers are too large, then
            # first two terms are merged.
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
    #print(all_ex, all_ey, all_ez)

    ef.close()
    return mode, mode_result, omega, False


#@jit(nopython=True, parallel=True)  # Set "nopython" mode for best performance, equivalent to @njit
# for some reason np.dot with complex number doesn't work with numba..
# could speed this up by making something in here parallel, but maybe not necessary
def calc_proj(all_ex, all_ey, all_ez, exp_array, n_kvecs, type_indices):
    # calculate projections
    mode_result = np.zeros(n_kvecs)  # length of number of k-vectors
    # sum over alpha
    for eigenvectors in [all_ex, all_ey, all_ez]:
        # multiply by exponential and sum

        # now sum over the the unit cell atoms
        for atom in range(Z):
            
            # get eigenvectors of this atom type
            eigs = eigenvectors[type_indices[atom]]
            
            # this dot product includes our sum over cells
            amplitude = np.dot(exp_array, eigs.T)  # should result in array of size (len(kvecs))
                
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
    testtile = np.zeros((3, 6), dtype='object')
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
    names = ['modes', 'projections', 'kvecs', 'omegas']
    objects = extend_projections(kvecs, np.array(all_modes), np.array(all_projections), np.array(all_omegas))
    for ii, obj in enumerate(objects):
        print(names[ii], obj)
        with h5py.File(os.path.join(outdir, '{}.h5'.format(names[ii])), 'w') as hf:
            hf.create_dataset("{}".format(names[ii]), data=obj)


#@profile
def project(k_vecs):

    # get cell coordinates
    with open(cellvecs_file, 'rb') as handle:
        cell_vecs = pickle.load(handle)

    # get exp_array
    exp_array = []
    for vecs in cell_vecs:
        exp, kvecs = make_exponential(N, vecs, k_vecs)
        exp_array.append(exp)

    # calculate maximum mode we need to project
    max_mode = calculate_max_mode(pfile, N, Z, omegamax)
    # option to change it to manual cutoff
    if manual_mode_cutoff > 0:
        max_mode = manual_mode_cutoff + 1

    # make type indices
    type_indices = make_indices(N**3, Z)

    # parallel loop over modes
    results = Parallel(n_jobs=nslots)(delayed(read_and_project_mode)(m, N, Z, exp_array, kvecs, type_indices) for m in range(mode_start, max_mode))
    all_modes = [x[0] for x in results]
    all_projections = [x[1] for x in results]
    all_omegas = [x[2] for x in results]
    save_projections(kvecs, all_modes, all_projections, all_omegas, outdir)
    return



project(k_vecs)

