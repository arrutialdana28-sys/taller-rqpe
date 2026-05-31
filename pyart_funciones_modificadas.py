#!/home/mrugna/mambaforge/envs/py311_rqpe/bin/python
#/usr/bin/env python3
"""
Created on Thu Aug 17 12:16:16 2023

@author: ushio_mac

Funciones incluidas

calculate_attenuation
phase_proc_lp
get_phidp_unf
unwrap_masked
construct_A_matrix
construct_B_vectors
LP_solver_pyglpk
"""

from copy import deepcopy
import copy
from warnings import warn
from time import time

import numpy as np
from numpy import ma
from scipy.integrate import cumulative_trapezoid as cumtrapz
import scipy.ndimage

from pyart.config import get_field_name, get_fillvalue, get_metadata
from pyart.filters import GateFilter, iso0_based_gate_filter, temp_based_gate_filter
from pyart.retrieve import get_freq_band
from pyart.correct.phase_proc import det_process_range, smooth_and_trim, smooth_masked


def calculate_attenuation(
    radar,
    z_offset,
    debug=False,
    doc=15,
    fzl=4000.0,
    gatefilter=None,
    rhv_min=0.8,
    ncp_min=0.5,
    a_coef=0.06,
    beta=0.8,
    refl_field=None,
    ncp_field=None,
    rhv_field=None,
    phidp_field=None,
    spec_at_field=None,
    corr_refl_field=None,
):
    """
    Calculate the attenuation from a polarimetric radar using Z-PHI method.

    Parameters
    ----------
    radar : Radar
        Radar object to use for attenuation calculations. Must have
        copol_coeff, norm_coherent_power, proc_dp_phase_shift,
        reflectivity_horizontal fields.
    z_offset : float
        Horizontal reflectivity offset in dBZ.
    debug : bool, optional
        True to print debugging information, False supressed this printing.
    doc : float, optional
        Number of gates at the end of each ray to to remove from the
        calculation.
    fzl : float, optional
        Freezing layer, gates above this point are not included in the
        correction.
    gatefilter : GateFilter, optional
        The gates to exclude from the calculation. This, combined with
        the gates above fzl, will be excluded from the correction. Set to
        None to not use a gatefilter.
    rhv_min : float, optional
        Minimum copol_coeff value to consider valid.
    ncp_min : float, optional
        Minimum norm_coherent_power to consider valid.
    a_coef : float, optional
        A coefficient in attenuation calculation.
    beta : float, optional
        Beta parameter in attenuation calculation.
    refl_field : str, optional
        Name of the reflectivity field used for the attenuation correction.
        A value of None for any of these parameters will use the default
        field name as defined in the Py-ART configuration file.
    phidp_field : str, optional
        Name of the differential phase field used for the attenuation
        correction. A value of None for any of these parameters will use the
        default field name as defined in the Py-ART configuration file.
    ncp_field : str, optional
        Name of the normalized coherent power field used for the attenuation
        correction. A value of None for any of these parameters will use the
        default field name as defined in the Py-ART configuration file.
    zdr_field : str, optional
        Name of the differential reflectivity field used for the attenuation
        correction. A value of None for any of these parameters will use the
        default field name as defined in the Py-ART configuration file. This
        will only be used if it is available.
    spec_at_field : str, optional
        Name of the specific attenuation field that will be used to fill in
        the metadata for the returned fields. A value of None for any of these
        parameters will use the default field names as defined in the Py-ART
        configuration file.
    corr_refl_field : str, optional
        Name of the corrected reflectivity field that will be used to fill in
        the metadata for the returned fields. A value of None for any of these
        parameters will use the default field names as defined in the Py-ART
        configuration file.

    Returns
    -------
    spec_at : dict
        Field dictionary containing the specific attenuation.
    cor_z : dict
        Field dictionary containing the corrected reflectivity.

    References
    ----------
    Gu et al. Polarimetric Attenuation Correction in Heavy Rain at C Band,
    JAMC, 2011, 50, 39-58.

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name("reflectivity")
    if ncp_field is None:
        ncp_field = get_field_name("normalized_coherent_power")
    if rhv_field is None:
        rhv_field = get_field_name("cross_correlation_ratio")
    if phidp_field is None:
        # use corrrected_differential_phae or unfolded_differential_phase
        # fields if they are available, if not use differential_phase field
        phidp_field = get_field_name("corrected_differential_phase")
        if phidp_field not in radar.fields:
            phidp_field = get_field_name("unfolded_differential_phase")
        if phidp_field not in radar.fields:
            phidp_field = get_field_name("differential_phase")
    if spec_at_field is None:
        spec_at_field = get_field_name("specific_attenuation")
    if corr_refl_field is None:
        corr_refl_field = get_field_name("corrected_reflectivity")

    # Extract fields and parameters from radar
    reflectivity_horizontal = radar.fields[refl_field]["data"]
    proc_dp_phase_shift = radar.fields[phidp_field]["data"]
    nsweeps = int(radar.nsweeps)

    # Determine where the reflectivity is valid, mask out bad locations.

    if gatefilter is None:
        gatefilter = GateFilter(radar)

    # Filter out the invalid values and apply rho_hv and ncp corrections
    gatefilter.exclude_invalid(refl_field)
    gatefilter.exclude_below(rhv_field, rhv_min)
    gatefilter.exclude_below(ncp_field, ncp_min)

    # Assign the mask to a variable
    mask = gatefilter.gate_excluded

    # Apply this mask to the reflectivity field
    refl = np.ma.masked_where(mask, reflectivity_horizontal + z_offset)

    # calculate initial reflectivity correction and gate spacing (in km)
    init_refl_correct = refl
    dr = (radar.range["data"][1] - radar.range["data"][0]) / 1000.0

    # create array to hold specific attenuation and attenuation
    specific_atten = np.zeros(reflectivity_horizontal.shape, dtype="float32")
    atten = np.zeros(reflectivity_horizontal.shape, dtype="float32")

    for sweep in range(nsweeps):
        # loop over the sweeps
        if debug:
            print("Doing ", sweep)
        end_gate, start_ray, end_ray = det_process_range(radar, sweep, fzl, doc=doc)

        for i in range(start_ray, end_ray):
            # perform attenuation calculation on a single ray

            # extract the ray's phase shift and init. refl. correction
            ray_phase_shift = proc_dp_phase_shift[i, 0:end_gate]
            ray_init_refl = init_refl_correct[i, 0:end_gate]

            # perform calculation
            last_six_good = np.where(~mask[i, 0:end_gate])[0][-6:]
            phidp_max = np.median(ray_phase_shift[last_six_good]) - ray_phase_shift[0]
            sm_refl = smooth_and_trim(ray_init_refl, window_len=5)
            reflectivity_linear = 10.0 ** (0.1 * beta * sm_refl)
            self_cons_number = 10.0 ** (0.1 * beta * a_coef * phidp_max) - 1.0
            I_indef = cumtrapz(0.46 * beta * dr * reflectivity_linear[::-1])
            I_indef = np.append(I_indef, I_indef[-1])[::-1]

            # set the specific attenutation and attenuation
            specific_atten[i, 0:end_gate] = (
                reflectivity_linear
                * self_cons_number
                / (I_indef[0] + self_cons_number * I_indef)
            )

            atten[i, :-1] = cumtrapz(specific_atten[i, :]) * dr * 2.0
            atten[i, -1] = atten[i, -2]

    # prepare output field dictionaries
    spec_at = get_metadata(spec_at_field)
    spec_at["data"] = specific_atten
    spec_at["_FillValue"] = get_fillvalue()

    cor_z = get_metadata(corr_refl_field)
    cor_z["data"] = atten + reflectivity_horizontal + z_offset
    cor_z["data"].mask = init_refl_correct.mask
    cor_z["_FillValue"] = get_fillvalue()

    return spec_at, cor_z

def phase_proc_lp(
    radar,
    offset,
    debug=True,
    self_const=60000.0,
    low_z=10.0,
    high_z=53.0,
    min_phidp=0.01,
    min_ncp=0.5,
    min_rhv=0.8,
    fzl=4000.0,
    sys_phase=0.0,
    overide_sys_phase=False,
    nowrap=None,
    really_verbose=False,
    LP_solver="pyglpk",
    refl_field=None,
    ncp_field=None,
    rhv_field=None,
    phidp_field=None,
    kdp_field=None,
    unf_field=None,
    window_len=35,
    proc=1,
    coef=0.914,
):
    """
    Phase process using a LP method [1].

    Parameters
    ----------
    radar : Radar
        Input radar.
    offset : float
        Reflectivity offset in dBz.
    debug : bool, optional
        True to print debugging information.
    self_const : float, optional
        Self consistency factor.
    low_z : float, optional
        Low limit for reflectivity. Reflectivity below this value is set to
        this limit.
    high_z : float, optional
        High limit for reflectivity. Reflectivity above this value is set to
        this limit.
    min_phidp : float, optional
        Minimum Phi differential phase.
    min_ncp : float, optional
        Minimum normal coherent power.
    min_rhv : float, optional
        Minimum copolar coefficient.
    fzl : float, optional
        Maximum altitude.
    sys_phase : float, optional
        System phase in degrees.
    overide_sys_phase : bool, optional
        True to use `sys_phase` as the system phase. False will calculate a
        value automatically.
    nowrap : int or None, optional
        Gate number to begin phase unwrapping. None will unwrap all phases.
    really_verbose : bool, optional
        True to print LPX messaging. False to suppress.
    LP_solver : 'pyglpk' or 'cvxopt', 'cylp', or 'cylp_mp', optional
        Module to use to solve LP problem. Default is 'pyglpk'.
    refl_field, ncp_field, rhv_field, phidp_field, kdp_field : str, optional
        Name of field in radar which contains the horizonal reflectivity,
        normal coherent power, copolar coefficient, differential phase shift,
        and differential phase. A value of None for any of these parameters
        will use the default field name as defined in the Py-ART configuration
        file.
    unf_field : str, optional
        Name of field which will be added to the radar object which will
        contain the unfolded differential phase. Metadata for this field
        will be taken from the phidp_field. A value of None will use
        the default field name as defined in the Py-ART configuration file.
    window_len : int, optional
        Length of Sobel window applied to PhiDP field when prior to
        calculating KDP.
    proc : int, optional
        Number of worker processes, only used when `LP_solver` is 'cylp_mp'.
    coef : float, optional
        Exponent linking Z to KDP in self consistency. kdp=(10**(0.1z))*coef

    Returns
    -------
    reproc_phase : dict
        Field dictionary containing processed differential phase shifts.
    sob_kdp : dict
        Field dictionary containing recalculated differential phases.

    References
    ----------
    [1] Giangrande, S.E., R. McGraw, and L. Lei. An Application of
    Linear Programming to Polarimetric Radar Differential Phase Processing.
    J. Atmos. and Oceanic Tech, 2013, 30, 1716.

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name("reflectivity")
    if ncp_field is None:
        ncp_field = get_field_name("normalized_coherent_power")
    if rhv_field is None:
        rhv_field = get_field_name("cross_correlation_ratio")
    if phidp_field is None:
        phidp_field = get_field_name("differential_phase")
    if kdp_field is None:
        kdp_field = get_field_name("specific_differential_phase")
    if unf_field is None:
        unf_field = get_field_name("unfolded_differential_phase")
        
    print('Entró phase_proc_lp')

    # prepare reflectivity field
    refl = copy.deepcopy(radar.fields[refl_field]["data"]) + offset
    is_low_z = (refl) < low_z
    is_high_z = (refl) > high_z
    refl[np.where(is_high_z)] = high_z
    refl[np.where(is_low_z)] = low_z
    z_mod = refl

    # unfold Phi_DP
    if debug:
        print("Unfolding")
    my_unf = get_phidp_unf(
        radar,
        ncp_lev=min_ncp,
        rhohv_lev=min_rhv,
        debug=debug,
        ncpts=2,
        doc=None,
        sys_phase=sys_phase,
        nowrap=nowrap,
        overide_sys_phase=overide_sys_phase,
        refl_field=refl_field,
        ncp_field=ncp_field,
        rhv_field=rhv_field,
        phidp_field=phidp_field,
    )
    my_new_ph = copy.deepcopy(radar.fields[phidp_field])
    my_unf[:, -1] = my_unf[:, -2]
    my_new_ph["data"] = my_unf
    radar.fields.update({unf_field: my_new_ph})

    phidp_mod = copy.deepcopy(radar.fields[unf_field]["data"])
    phidp_neg = phidp_mod < min_phidp
    phidp_mod[np.where(phidp_neg)] = min_phidp

    # process
    proc_ph = copy.deepcopy(radar.fields[phidp_field])
    proc_ph["data"] = phidp_mod
    St_Gorlv_differential_5pts = [-0.2, -0.1, 0, 0.1, 0.2]
    for sweep in range(len(radar.sweep_start_ray_index["data"])):
        if debug:
            print("Doing ", sweep)
        end_gate, start_ray, end_ray = det_process_range(radar, sweep, fzl, doc=15)
        start_gate = 0

        A_Matrix = construct_A_matrix(
            len(radar.range["data"][start_gate:end_gate]), St_Gorlv_differential_5pts
        )

        B_vectors = construct_B_vectors(
            phidp_mod[start_ray:end_ray, start_gate:end_gate],
            z_mod[start_ray:end_ray, start_gate:end_gate],
            St_Gorlv_differential_5pts,
            dweight=self_const,
            coef=coef,
        )

        weights = np.ones(phidp_mod[start_ray:end_ray, start_gate:end_gate].shape)

        nw = np.bmat([weights, np.zeros(weights.shape)])

        if LP_solver == "pyglpk":
            mysoln = LP_solver_pyglpk(
                A_Matrix, B_vectors, nw, really_verbose=really_verbose
            )
        else:
            raise ValueError("unknown LP_solver:" + LP_solver)

        proc_ph["data"][start_ray:end_ray, start_gate:end_gate] = mysoln

    last_gates = proc_ph["data"][start_ray:end_ray, -16]
    proc_ph["data"][start_ray:end_ray, -16:] = np.meshgrid(np.ones([16]), last_gates)[1]
    proc_ph["valid_min"] = 0.0  # XXX is this correct?
    proc_ph["valid_max"] = 400.0  # XXX is this correct?

    # prepare output
    sobel = 2.0 * np.arange(window_len) / (window_len - 1.0) - 1.0
    sobel = sobel / (abs(sobel).sum())
    sobel = sobel[::-1]
    gate_spacing = (radar.range["data"][1] - radar.range["data"][0]) / 1000.0
    kdp = scipy.ndimage.convolve1d(proc_ph["data"], sobel, axis=1) / (
        (window_len / 3.0) * 2.0 * gate_spacing
    )

    # copy the KDP metadata from existing field or create anew
    if kdp_field in radar.fields:
        sob_kdp = copy.deepcopy(radar.fields[kdp_field])
    else:
        sob_kdp = get_metadata(kdp_field)

    sob_kdp["data"] = kdp
    sob_kdp["_FillValue"] = get_fillvalue()

    return proc_ph, sob_kdp

def get_phidp_unf(
    radar,
    ncp_lev=0.4,
    rhohv_lev=0.6,
    debug=False,
    ncpts=20,
    doc=-10,
    overide_sys_phase=False,
    sys_phase=-135,
    nowrap=None,
    refl_field=None,
    ncp_field=None,
    rhv_field=None,
    phidp_field=None,
):
    """
    Get Unfolded Phi differential phase

    Parameters
    ----------
    radar : Radar
        The input radar.
    ncp_lev : float, optional
        Miminum normal coherent power level. Regions below this value will
        not be included in the calculation.
    rhohv_lev : float, optional
        Miminum copolar coefficient level. Regions below this value will not
        be included in the calculation.
    debug : bool, optional
        True to print debugging information, False to supress printing.
    ncpts : int, optional
        Minimum number of points in a ray. Regions within a ray smaller than
        this or beginning before this gate number are excluded from
        calculations.
    doc : int or None, optional
        Index of first gate not to include in field data, None include all.
    overide_sys_phase : bool, optional
        True to use `sys_phase` as the system phase. False will determine a
        value automatically.
    sys_phase : float, optional
        System phase, not used if overide_sys_phase is False.
    nowrap : int or None, optional
        Gate number where unwrapping should begin. `None` will unwrap all
        gates.
    refl_field ncp_field, rhv_field, phidp_field : str, optional
        Field names within the radar object which represent the horizonal
        reflectivity, normal coherent power, the copolar coefficient, and the
        differential phase shift. A value of None for any of these parameters
        will use the default field name as defined in the Py-ART
        configuration file.

    Returns
    -------
    cordata : array
        Unwrapped phi differential phase.

    """
    # parse the field parameters
    if refl_field is None:
        refl_field = get_field_name("reflectivity")
    if ncp_field is None:
        ncp_field = get_field_name("normalized_coherent_power")
    if rhv_field is None:
        rhv_field = get_field_name("cross_correlation_ratio")
    if phidp_field is None:
        phidp_field = get_field_name("differential_phase")

    if doc is not None:
        my_phidp = radar.fields[phidp_field]["data"][:, 0:doc]
        my_rhv = radar.fields[rhv_field]["data"][:, 0:doc]
        my_ncp = radar.fields[ncp_field]["data"][:, 0:doc]
        my_z = radar.fields[refl_field]["data"][:, 0:doc]
    else:
        my_phidp = radar.fields[phidp_field]["data"]
        my_rhv = radar.fields[rhv_field]["data"]
        my_ncp = radar.fields[ncp_field]["data"]
        my_z = radar.fields[refl_field]["data"]
    t = time()
    if overide_sys_phase:
        system_zero = sys_phase
    cordata = np.zeros(my_rhv.shape, dtype=float)
    for radial in range(my_rhv.shape[0]):
        notmeteo = np.logical_or(my_ncp[radial, :] < ncp_lev, my_rhv[radial, :] < rhohv_lev)
        x_ma = ma.masked_where(notmeteo, my_phidp[radial, :])
        if nowrap is not None:
            # Start the unfolding a bit later in order to avoid false
            # jumps based on clutter
            unwrapped = copy.deepcopy(x_ma)
            end_unwrap = unwrap_masked(x_ma[nowrap::], centered=False)
            unwrapped[nowrap::] = end_unwrap
        else:
            unwrapped = unwrap_masked(x_ma, centered=False)
        # end so no clutter expected
        system_max = (
            unwrapped[np.where(np.logical_not(notmeteo))][-10:-1].mean() - system_zero
        )
        unwrapped_fixed = np.zeros(len(x_ma), dtype=float)
        based = unwrapped - system_zero
        based[0] = 0.0
        notmeteo[0] = False
        based[-1] = system_max
        notmeteo[-1] = False
        unwrapped_fixed[np.where(np.logical_not(based.mask))[0]] = based[
            np.where(np.logical_not(based.mask))[0]
        ]
        if len(based[np.where(np.logical_not(based.mask))[0]]) > 11:
            unwrapped_fixed[np.where(based.mask)[0]] = np.interp(
                np.where(based.mask)[0],
                np.where(np.logical_not(based.mask))[0],
                smooth_and_trim(based[np.where(np.logical_not(based.mask))[0]]),
            )
        else:
            unwrapped_fixed[np.where(based.mask)[0]] = np.interp(
                np.where(based.mask)[0],
                np.where(np.logical_not(based.mask))[0],
                based[np.where(np.logical_not(based.mask))[0]],
            )
        c=0
        if c != 1:
            cordata[radial, :] = unwrapped_fixed
        else:
            cordata[radial, :] = np.zeros(my_rhv.shape[1])
    if debug:
        print("Exec time: ", time() - t)
    return cordata

def unwrap_masked(lon, centered=False, copy=True):
    """
    Unwrap a sequence of longitudes or headings in degrees.

    Parameters
    ----------
    lon : array
        Longtiudes or heading in degress. If masked output will also be
        masked.
    centered : bool, optional
        Center the unwrapping as close to zero as possible.
    copy : bool, optional.
        True to return a copy, False will avoid a copy when possible.

    Returns
    -------
    unwrap : array
        Array of unwrapped longtitudes or headings, in degrees.

    """
    masked_input = ma.isMaskedArray(lon)
    if masked_input:
        fill_value = lon.fill_value
        # masked_invalid loses the original fill_value (ma bug, 2011/01/20)
    lon = np.ma.masked_invalid(lon).astype(float)
    if lon.ndim != 1:
        raise ValueError("Only 1-D sequences are supported")
    if lon.shape[0] < 2:
        return lon
    x = lon.compressed()
    if len(x) < 2:
        return lon
    w = np.zeros(x.shape[0] - 1, int)
    ld = np.diff(x)
    np.putmask(w, ld > 180, -1)
    np.putmask(w, ld < -180, 1)
    x[1:] += w.cumsum() * 360.0
    if centered:
        x -= 360 * np.round(x.mean() / 360.0)
    if lon.mask is ma.nomask:
        lon[:] = x
    else:
        lon[~lon.mask] = x
    if masked_input:
        lon.fill_value = fill_value
        return lon
    else:
        return lon.filled(np.nan)
    
def construct_A_matrix(n_gates, filt):
    """
    Construct a row-augmented A matrix. Equation 5 in Giangrande et al, 2012.

    A is a block matrix given by:

    .. math::

        \\bf{A} = \\begin{bmatrix} \\bf{I} & \\bf{-I} \\\\\\\\
                  \\bf{-I} & \\bf{I} \\\\\\\\ \\bf{Z}
                  & \\bf{M} \\end{bmatrix}

    where
        :math:`\\bf{I}` is the identity matrix
        :math:`\\bf{Z}` is a matrix of zeros
        :math:`\\bf{M}` contains our differential constraints.

    Each block is of shape n_gates by n_gates making
    shape(:math:`\\bf{A}`) = (3 * n, 2 * n).

    Note that :math:`\\bf{M}` contains some side padding to deal with edge
    issues.

    Parameters
    ----------
    n_gates : int
        Number of gates, determines size of identity matrix.
    filt : array
        Input filter.

    Returns
    -------
    a : matrix
        Row-augmented A matrix.

    """
    Identity = np.eye(n_gates)
    filter_length = len(filt)
    M_matrix_middle = np.diag(np.ones(n_gates - filter_length + 1), k=0) * 0.0
    posn = np.linspace(
        -1.0 * (filter_length - 1) / 2, (filter_length - 1) / 2, filter_length
    )
    for diag in range(filter_length):
        M_matrix_middle = (
            M_matrix_middle
            + np.diag(
                np.ones(int(n_gates - filter_length + 1 - np.abs(posn[diag]))),
                k=int(posn[diag]),
            )
            * filt[diag]
        )
    side_pad = (filter_length - 1) // 2
    M_matrix = np.bmat(
        [
            np.zeros([n_gates - filter_length + 1, side_pad], dtype=float),
            M_matrix_middle,
            np.zeros([n_gates - filter_length + 1, side_pad], dtype=float),
        ]
    )
    Z_matrix = np.zeros([n_gates - filter_length + 1, n_gates])
    return np.bmat(
        [[Identity, -1.0 * Identity], [Identity, Identity], [Z_matrix, M_matrix]]
    )


def construct_B_vectors(phidp_mod, z_mod, filt, coef=0.914, dweight=60000.0):
    """
    Construct B vectors. See Giangrande et al, 2012.

    Parameters
    ----------
    phidp_mod : 2D array
        Phi differential phases.
    z_mod : 2D array.
       Reflectivity, modified as needed.
    filt : array
        Input filter.
    coef : float, optional.
        Cost coefficients.
    dweight : float, optional.
        Weights.

    Returns
    -------
    b : matrix
        Matrix containing B vectors.

    """
    n_gates = phidp_mod.shape[1]
    n_rays = phidp_mod.shape[0]
    filter_length = len(filt)
    side_pad = (filter_length - 1) // 2
    top_of_B_vectors = np.bmat([[-phidp_mod, phidp_mod]])
    data_edges = np.bmat(
        [
            phidp_mod[:, 0:side_pad],
            np.zeros([n_rays, n_gates - filter_length + 1]),
            phidp_mod[:, -side_pad:],
        ]
    )
    ii = filter_length - 1
    jj = data_edges.shape[1] - 1
    list_corrl = np.zeros([n_rays, jj - ii + 1])
    for count in range(list_corrl.shape[1]):
        list_corrl[:, count] = -1.0 * (
            np.array(filt) * (np.asarray(data_edges))[:, count : count + ii + 1]
        ).sum(axis=1)

    sct = ((10.0 ** (0.1 * z_mod)) ** coef / dweight)[:, side_pad:-side_pad]
    sct[np.where(sct < 0.0)] = 0.0
    sct[:, 0:side_pad] = list_corrl[:, 0:side_pad]
    sct[:, -side_pad:] = list_corrl[:, -side_pad:]
    B_vectors = np.bmat([[top_of_B_vectors, sct]])
    return B_vectors

def LP_solver_pyglpk(
    A_Matrix, B_vectors, weights, it_lim=7000, presolve=True, really_verbose=False
):
    """
    Solve the Linear Programming problem given in Giangrande et al, 2012 using
    the PyGLPK module.

    Parameters
    ----------
    A_Matrix : matrix
        Row augmented A matrix, see :py:func:`construct_A_matrix`
    B_vectors : matrix
        Matrix containing B vectors, see :py:func:`construct_B_vectors`
    weights : array
        Weights.
    it_lim : int, optional
        Simplex iteration limit.
    presolve : bool, optional
        True to use the LP presolver.
    really_verbose : bool, optional
        True to print LPX messaging. False to suppress.

    Returns
    -------
    soln : array
        Solution to LP problem.

    See Also
    --------
    LP_solver_cvxopt : Solve LP problem using the CVXOPT module.
    LP_solver_cylp : Solve LP problem using the cylp module.
    LP_solver_cylp_mp : Solve LP problem using the cylp module
                        using multi processes.

    """
    import glpk

    if really_verbose:
        message_state = glpk.LPX.MSG_ON
    else:
        message_state = glpk.LPX.MSG_OFF
    n_gates = weights.shape[1] // 2
    n_rays = B_vectors.shape[0]
    mysoln = np.zeros([n_rays, n_gates])
    lp = glpk.LPX()  # Create empty problem instance
    lp.name = "LP_MIN"  # Assign symbolic name to problem
    lp.obj.maximize = False  # Set this as a maximization problem
    lp.rows.add(2 * n_gates + n_gates - 4)  # Append rows
    lp.cols.add(2 * n_gates)
    glpk.env.term_on = True
    for cur_row in range(2 * n_gates + n_gates - 4):
        lp.rows[cur_row].matrix = list(np.squeeze(np.asarray(A_Matrix[cur_row, :])))
    for i in range(2 * n_gates):
        lp.cols[i].bounds = 0.0, None
    for raynum in range(n_rays):
        this_soln = np.zeros(n_gates)
        for i in range(2 * n_gates + n_gates - 4):
            lp.rows[i].bounds = B_vectors[raynum, i], None
        for i in range(2 * n_gates):
            lp.obj[i] = weights[raynum, i]
        lp.simplex(
            msg_lev=message_state,
            meth=glpk.LPX.PRIMAL,
            it_lim=it_lim,
            presolve=presolve,
        )
        for i in range(n_gates):
            this_soln[i] = lp.cols[i + n_gates].primal
        mysoln[raynum, :] = smooth_and_trim(this_soln, window_len=5, window="sg_smooth")
    return mysoln
