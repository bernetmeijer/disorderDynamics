"""
A few functions that help convert output of projection to a Euphonic Spectrum 2D object, which will create pretty plots.
"""

import numpy as np
import pandas as pd
from typing import Optional, Sequence, Tuple, Union
from euphonic import Quantity
from euphonic import Spectrum2D
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, Normalize
from matplotlib.image import NonUniformImage


# binning function
def binning(df, ymax, ymin, n_bins):
    kplot = df['kplot'].to_numpy()
    om = df['omega'].to_numpy()
    proj = df['projection'].to_numpy()

    binned_k = []
    binned_om = []
    binned_proj = []
    for kk in np.unique(kplot):
        kslice = np.where(kplot == kk)[0]
        komegas = om[kslice]
        kprojection = proj[kslice]
        hist, bin_edges = np.histogram(komegas, bins=n_bins, weights=kprojection, range=(ymin, ymax))

        binned_om.append((bin_edges[:-1] + (bin_edges[1] - bin_edges[0])))
        binned_proj.append(hist)
        binned_k.append(np.ones(len(hist)) * kk)

    binned_k = np.concatenate(binned_k)
    binned_om = np.concatenate(binned_om)
    binned_proj = np.concatenate(binned_proj)

    df2 = pd.DataFrame({'kplot': binned_k, 'omega': binned_om, 'projection': binned_proj})

    return df2


def df_to_spectrum2D(df, ymax, ymin, n_bins):
    kplot = df['kplot'].to_numpy()
    om = df['omega'].to_numpy()
    proj = df['projection'].to_numpy()

    binned_k = []
    binned_om = []
    binned_proj = []

    kmax = max(np.unique(kplot))
    for kk in range(0, kmax + 1):
        kslice = np.where(kplot == kk)[0]
        komegas = om[kslice]
        kprojection = proj[kslice]
        hist, bin_edges = np.histogram(komegas, bins=n_bins, weights=kprojection, range=(ymin, ymax))

        binned_om.append((bin_edges[:-1] + (bin_edges[1] - bin_edges[0])))
        binned_proj.append(hist)
        binned_k.append(kk)

    x_data = np.array(binned_k)
    y_data = binned_om[0]
    z_data = np.array(binned_proj)

    mydict = {'x_data': x_data, 'y_data': y_data, 'z_data': z_data}

    return mydict


def xrange_into_plot(xrange, segment_lengths, seglen=20, label_locs=None):
    xstart = 0
    xplots = []

    if label_locs is not None:
        for ii, segment in enumerate(segment_lengths):
            if ii == 0:
                start = label_locs[ii]
            else:
                start = label_locs[ii] + 1
            end = label_locs[ii + 1] + 1
            x = xrange[start:end]
            # print(start, end, x)

            if ii == 0:
                stepsize = segment / (len(x) - 1)
                xplot = np.arange(0, len(x)) * stepsize + xstart
            else:
                stepsize = segment / len(x)
                xplot = np.arange(1, len(x) + 1) * stepsize + xstart

            xplots += list(xplot)
            # print(list(xplot))
            # xplots.append(xplot)

            xstart = xplot[-1]

        return np.array(xplots)

    for ii, segment in enumerate(segment_lengths):
        if ii == (len(segment_lengths) - 1):
            end = (ii + 1) * seglen
            start = (seglen * ii + 1)
        elif ii == 0:
            end = (ii + 1) * seglen + 1
            start = (seglen * ii)
        else:
            end = (ii + 1) * seglen + 1
            start = (seglen * ii + 1)

        x = xrange[start:end]
        # print(start,end, x)

        stepsize = segment / len(x)

        xplot = np.arange(1, len(x) + 1) * stepsize + xstart
        xplots += list(xplot)
        # print(xplots)

        xstart = xplot[-1]

    return np.array(xplots)


def frequencies_to_spectrum2D(qpointfrequencies, kdict, segment_lengths, ymin, ymax, n_bins):
    """
    This will turn a euphonic QPointFrequencies object into a Spectrum2D object.
    This is useful if you want to make an image of the dispersion curve without neutron weighting.
    Parameters
    ----------
    qpointfrequencies: Euphonc QPointFrequencies object
    kdict: dictionary
        has two items, 'labels' and 'label_positions'.
        labels are the labels of high-symmetry directions
        and their locations are the **number** of the q-point at which they occur
        (so not q-pt coordinate!), starting at 0 and ends at qpointfrequencies.n_qpts
    segment_lengths: list of integers
        for each segment (between two high-symmetry points, so corresponding to the label positions
        in kdict), how long you want the segment to be in the plot
    ymin: float
        minimum energy
    ymax: float
        maximum energy
    n_bins: integer
        how many bins in your energy binning

    Returns
    -------
    Euphonic Spectrum2D object
    """

    alldata = []
    for kk in range(0, qpointfrequencies.n_qpts):
        fs = qpointfrequencies.frequencies.magnitude[kk]
        for f in fs:
            alldata.append([kk, f, 1])  # kplot, frequency, projection=1

    df = pd.DataFrame(alldata, columns=['kplot', 'omega', 'projection'])

    # make dictionary to turn into spectrum2D format
    mydict = df_to_spectrum2D(df, ymax, ymin, n_bins)
    print(mydict['x_data'])
    mydict['x_data'] = xrange_into_plot(mydict['x_data'], segment_lengths,
                                        label_locs=kdict['label_positions'])

    mydict['x_data_unit'] = '1 / angstrom'
    mydict['y_data_unit'] = 'millielectron_volt'
    mydict['z_data_unit'] = 'angstrom ** 2 / millielectron_volt'
    # make x tick labels list
    x_tick_labels = []
    for ll, label in enumerate(kdict['labels']):
        mytuple = tuple([kdict['label_positions'][ll], label])
        x_tick_labels.append(mytuple)
    mydict['x_tick_labels'] = x_tick_labels

    return mydict


# the two functions below are copied exactly from euphonic (so could also load euphonic instead)
def plot_2d_to_axis(spectrum: Spectrum2D, ax: Axes,
                    cmap: Union[str, Colormap, None] = None,
                    interpolation: str = 'nearest',
                    norm: Optional[Normalize] = None,
                    ) -> NonUniformImage:
    """Plot Spectrum2D object to Axes
    Parameters
    ----------
    spectrum
        2D data object for plotting as NonUniformImage. The x_tick_labels
        attribute will be used to mark labelled points.
    ax
        Matplotlib axes to which image will be drawn
    cmap
        Matplotlib colormap or registered colormap name
    interpolation
        Interpolation method: 'nearest' or 'bilinear' for a pixellated or
        smooth result
    norm
        Matplotlib normalization object; set this in order to ensure separate
        plots are on the same colour scale.
    """
    x_unit = spectrum.x_data_unit
    y_unit = spectrum.y_data_unit
    z_unit = spectrum.z_data_unit

    x_bins = spectrum.get_bin_edges('x').to(x_unit).magnitude
    y_bins = spectrum.get_bin_edges('y').to(y_unit).magnitude

    image = NonUniformImage(ax, interpolation=interpolation,
                            extent=(min(x_bins), max(x_bins),
                                    min(y_bins), max(y_bins)),
                            cmap=cmap)
    if norm is not None:
        image.set_norm(norm)

    image.set_data(spectrum.get_bin_centres('x').to(x_unit).magnitude,
                   spectrum.get_bin_centres('y').to(y_unit).magnitude,
                   spectrum.z_data.to(z_unit).magnitude.T)
    ax.add_image(image)
    ax.set_xlim(min(x_bins), max(x_bins))
    ax.set_ylim(min(y_bins), max(y_bins))

    _set_x_tick_labels(ax, spectrum.x_tick_labels, spectrum.x_data)

    return image


def _set_x_tick_labels(ax: Axes,
                       x_tick_labels: Optional[Sequence[Tuple[int, str]]],
                       x_data: Quantity) -> None:
    if x_tick_labels is not None:
        locs, labels = [list(x) for x in zip(*x_tick_labels)]
        x_values = x_data.magnitude  # type: np.ndarray
        ax.set_xticks(x_values[locs])

        # Rotate long tick labels
        if len(max(labels, key=len)) >= 11:
            ax.set_xticklabels(labels, rotation=90)
        else:
            ax.set_xticklabels(labels)

    # set vertical lines
    for xc in locs:
        ax.axvline(x=x_values[xc], color='grey', linewidth=3)
