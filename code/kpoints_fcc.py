# python script containing function to calculate the allowed k-points
# in the FCC supercell
# this code is based on the paper by Boykin 2006
# you should end up with 4 * N**3 modes in an NxNxN supercell (equals the number of primitive cells in the supercell)

# in all of the following, we ignore the factor 2pi/a which appears in all k vectors.
# here a is the length of the unit cell side (same for all sides, since we're in cubic setting)

import numpy as np

""" Some FCC-particular settings """

# FCC additional vectors
fcc_vecs = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

# Brillouin zone faces
Xpoints = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
Lpoints = [0.5*np.array([1, 1, 1]), 0.5*np.array([-1, 1, 1]), 0.5*np.array([1, -1, 1]), 0.5*np.array([1, 1, -1]),
           0.5*np.array([-1, -1, 1]), 0.5*np.array([1, -1, -1]), 0.5*np.array([-1, 1, -1]), 0.5*np.array([-1, -1, -1])]
# all brillouin zone boundary points
Bpoints = np.concatenate((Xpoints, np.array(Lpoints)), axis=0)


def supercell_vectors(N):
    """
    Parameters
    ----------
    N: supercell size (same in each direction)

    Returns
    -------
    k vector according to standard supercell rules
    """
    Nbig= float(N)
    super_k = []
    for nx in range(N):
        for ny in range(N):
            for nz in range(N):
                super_k.append([nx/Nbig, ny/Nbig, nz/Nbig])
    return super_k


def all_allowed_k(N):
    all_k = []
    super_k = supercell_vectors(N)
    for superk in super_k:
        for fcc_vec in fcc_vecs:
            all_k.append(np.array(superk)+ np.array(fcc_vec))
    return all_k


def shift_k(k, Bpoints):
    k = np.array(k)
    for xp in Bpoints:
        n = xp/np.linalg.norm(xp)
        zonecheck = np.dot((k - xp), n)
        if zonecheck > 0:
            #print('moving k back into brillouin zone')
            k = k-xp
            # check if this is the right method
            zonecheck = np.dot((k - xp), n)
            if zonecheck > 0:
                print('still outside brillouin zone!!')
    return k


def all_k_in_zone(N):
    allk = all_allowed_k(N)
    shifted_k = []
    for k in allk:
        shifted_k.append(shift_k(k, Bpoints))
    return shifted_k
