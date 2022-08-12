import os
import pandas as pd
import numpy as np


def get_locations(phononfile, outdir):
    """From castep file, get locations of all atoms and put them in a dataframe.
    """
    alldata = []
    coordstart = 'no'
    collected_ions = 0
    n_ions = 1000

    with open(phononfile, 'r') as ef:
        for i, line in enumerate(ef):
            terms = line.split()
            if len(terms) == 0:
                continue

            # get number of ions
            if terms[0] == 'Number' and terms[2] == 'ions':
                n_ions = int(terms[3])

            # where coordinates start
            elif terms[0] == 'Fractional':
                coordstart = 'go'
                continue

            # get coordinates
            elif coordstart == 'go' and collected_ions < n_ions:
                ion = terms[0]
                a = float(terms[1])
                b = float(terms[2])
                c = float(terms[3])
                alldata.append([ion, np.array([a, b, c])])
                collected_ions += 1

            elif collected_ions >= n_ions:

                # turn data in dataframe
                df = pd.DataFrame(alldata, columns=['atom_id', 'coord'])

                df.to_pickle(os.path.join(outdir, 'coordinates.pkl'))
                return
