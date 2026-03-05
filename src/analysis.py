"""analysis.py
Data reduction, z-scoring and statistical analysis
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import smBaseFunctions3 as sbf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import copy
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any, Callable, Mapping

logger = logging.getLogger(__name__)

################################################################################################################
def generate_mean_rate(
    allDat: Mapping[Any, Mapping[str, Any]],
    celltype: Iterable[str],
    xrange: Tuple[int, int] = (0, 150)
) -> Dict[Any, Dict[str, np.ndarray]]:
    """
    Compute the mean firing rate (ignoring NaNs) for specified cell types
    within a given index range.

    Parameters
    ----------
    allDat : Mapping[Any, Mapping[str, Any]]
        Nested dictionary-like structure where:
        - Outer keys represent dataset/session identifiers.
        - Inner keys correspond to cell types.
        - allDat[key][celltype][0] is expected to be an array-like object
          containing rate data.
    celltype : Iterable[str]
        Iterable of cell type names to process.
    xrange : Tuple[int, int], optional
        Start (inclusive) and end (exclusive) indices defining the slice
        over which to compute the mean. Default is (0, 150).

    Returns
    -------
    Dict[Any, Dict[str, np.ndarray]]
        Dictionary with the same outer keys as `allDat`. Each value is a
        dictionary mapping cell types to their mean rate (computed with
        `np.nanmean` over the specified range along axis 0).
    """
    odata: Dict[Any, Dict[str, np.ndarray]] = {}

    for ikey, val in allDat.items():
        tempdict: Dict[str, np.ndarray] = {}
        for ctype in celltype:
            tempdat = np.squeeze(val[ctype][0])
            slice_data = tempdat[xrange[0]:xrange[1]]
            tempdict[ctype] = np.nanmean(slice_data, axis=0)
        odata[ikey] = tempdict

    return odata
####################################################################################
def bin_array(
    idata: np.ndarray,
    axis: int = 0,
    binstep: int = 10,
    binsize: int = 10,
    func: Callable[[np.ndarray, int], np.ndarray] = np.nanmean
) -> np.ndarray:
    """
    Bin an array along a specified axis using a sliding window.

    Binning is performed by stepping through the specified axis in increments
    of `binstep` and applying `func` over windows of size `binsize`.

    Parameters
    ----------
    idata : np.ndarray
        Input array to be binned.
    axis : int, optional
        Axis along which to perform binning. Default is 0.
    binstep : int, optional
        Step size between consecutive bins. Default is 10.
    binsize : int, optional
        Size of each bin window. Default is 10.
    func : Callable[[np.ndarray, int], np.ndarray], optional
        Function applied to each bin. Must accept an array and an axis
        argument. Default is `np.nanmean`.

    Returns
    -------
    np.ndarray
        Binned array with the same number of dimensions as the input,
        where the specified axis has been reduced according to the
        binning procedure.
    """
    idata = np.asarray(idata)
    dims = np.array(idata.shape)

    # Move target axis to front
    argdims = np.arange(idata.ndim)
    argdims[0], argdims[axis] = argdims[axis], argdims[0]
    idata = idata.transpose(argdims)

    # Perform binning
    binned = [
        func(
            np.take(idata, np.arange(int(i * binstep), int(i * binstep + binsize)), axis=0),
            axis=0,
        )
        for i in range(dims[axis] // binstep)
    ]

    odata = np.array(binned).transpose(argdims)

    return odata  
##########################################################################################################################
def _ensure_2d_channel_axis(x: np.ndarray) -> np.ndarray:
    """
    Ensure array has shape (time, channels). If input is 1D, convert to (N, 1).
    If input already 2D, return as-is (but not a view).
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(-1, 1).copy()
    elif x.ndim == 2:
        return x.copy()
    else:
        # Try to squeeze singleton axes, then ensure 2D
        s = np.squeeze(x)
        if s.ndim == 1:
            return s.reshape(-1, 1).copy()
        elif s.ndim == 2:
            return s.copy()
        else:
            raise ValueError(f"Input array must be 1D or 2D after squeezing, got shape {x.shape}")
#######################################################################################################################
def generate_z_score(
    allDat: Dict,
    okey_list: Sequence[str],
    smpts: Optional[int] = 3,
    polyorder: int = 1,
) -> Dict:
    """
    Compute z-score per (time x channel) data for every entry in `allDat` and
    for each key in `okey_list`.

    Parameters
    ----------
    allDat : dict
        Mapping id -> dict-like that contains arrays under keys in `okey_list`.
        Expected access pattern in original code: allDat[id][ctype][0] is the data.
    okey_list : sequence of str
        List of keys to process inside each allDat[item].
    smpts : int, optional
        If provided, apply Savitzky-Golay smoothing with this window length (must be odd).
    polyorder : int
        Polynomial order for Savitzky-Golay (default 1).

    Returns
    -------
    Dict
        New dict with same top-level keys; values are dicts mapping each ctype to
        a z-scored numpy array with shape (time, channels).
    """
    from scipy.signal import savgol_filter
    if smpts is not None:
        if smpts <= polyorder:
            raise ValueError("smpts (window_length) must be > polyorder")
        if smpts % 2 == 0:
            raise ValueError("smpts (window_length) must be odd for savgol_filter")

    out = {}
    for ikey, val in allDat.items():
        tempdict = {}
        for ctype in okey_list:
            # gracefully get the data where original used val[ctype][0]
            try:
                raw = val[ctype][0]
            except Exception:
                # fallback if the structure is slightly different
                raw = val[ctype]

            arr = _ensure_2d_channel_axis(raw)

            if smpts is not None:
                # savgol_filter expects axis=0 if rows=time
                # window_length must be <= arr.shape[0]
                if smpts > arr.shape[0]:
                    raise ValueError(
                        f"smpts ({smpts}) larger than data length ({arr.shape[0]}) for key {ikey}/{ctype}"
                    )
                # apply filter along time axis
                arr_smoothed = savgol_filter(arr, window_length=smpts, polyorder=polyorder, axis=0)
            else:
                arr_smoothed = arr

            # zscore along time axis (axis=0) -> preserve time x channels layout
            # stats.zscore returns float64; replace constant-column nan with 0
            z = stats.zscore(arr_smoothed, axis=0, nan_policy="propagate")
            # If a channel is constant, zscore yields nan; replace those with 0 (or keep nan if preferred)
            const_mask = np.isnan(z).all(axis=0)
            if const_mask.any():
                # replace NaNs only in fully-NaN columns
                z[:, const_mask] = 0.0

            tempdict[ctype] = z
        out[ikey] = tempdict

    return out
####################################################################################################################
def remove_nans_paired(
    iDict: Dict,
    okey_list: Sequence[str],
    ikey_list: Tuple[str, str] = ("ds", "swr"),
    warn: bool = True,
) -> Dict:
    """
    For every okey in okey_list, remove channels (columns) where *either*
    iDict[ikey_list[0]][okey] or iDict[ikey_list[1]][okey] contains NaNs
    in that column. Works with arrays shaped (time, channels) or 1D arrays.

    This returns a new dict (a shallow copy of iDict with arrays replaced).
    """
    out = copy.deepcopy(iDict)

    a_key, b_key = ikey_list
    for okey in okey_list:
        a_raw = out[a_key][okey]
        b_raw = out[b_key][okey]

        a = _ensure_2d_channel_axis(a_raw)
        b = _ensure_2d_channel_axis(b_raw)

        if a.shape[0] != b.shape[0]:
            # time dimension mismatch is suspicious
            logger.warning(
                "Time dimension mismatch for key %s: %s timepoints vs %s timepoints. Proceeding anyway.",
                okey,
                a.shape[0],
                b.shape[0],
            )

        # mask = True for columns to keep (no NaNs in either a or b for that column)
        mask = ~np.logical_or(np.isnan(a).any(axis=0), np.isnan(b).any(axis=0))

        if not mask.any() and warn:
            logger.warning("No valid channels remain after NaN-pair removal for key %s", okey)

        # preserve shape (time, n_kept_channels). If nothing remains, shape (time,0)
        a_kept = a[:, mask]
        b_kept = b[:, mask]

        out[a_key][okey] = a_kept
        out[b_key][okey] = b_kept

    return out
###########################################################################################################
def drop_nan_for_df(idata,outerkeys):
    
    '''
    This function will remove nan values from a dict so that it can 
    be processed by dabest and other analysis functions
    '''
    
    odata = {}
    nanIndx = []
    for okeyindx,okey in enumerate(outerkeys): 
        nanIndx.append(idata[okey])
        
    nas = np.logical_or.reduce([np.isnan(x) for x in nanIndx])
    
    for okeyindx,okey in enumerate(outerkeys):
        odata[okey] = nanIndx[okeyindx][~nas]
        
    return odata
########################################################################
def nan_free_dict(idata,outerkeys,innerkeys):
    '''
    Function to create dict without any nan values
    returns odata[ctype]... with structure odata[ctype][sess]
    '''
    odata = {}
    for ikeyindx,ikey in enumerate(innerkeys):
        odata[ikey] = drop_nan_for_df(idata[ikey],outerkeys)
        
    return odata  
########################################################################
def generate_nan_free_z_score(
    allDat: Dict,
    okey_list: Sequence[str],
    ikey_list: Tuple[str, str] = ("ds", "swr"),
    smpts: Optional[int] = None,
    polyorder: int = 1,
) -> Dict:
    """
    Helper that computes z-scores and then removes paired NaN channels.
    Returns a new nan-free dict.
    """
    zdict = generate_z_score(allDat, okey_list, smpts=smpts, polyorder=polyorder)
    clean = remove_nans_paired(zdict, okey_list, ikey_list=ikey_list)

    return clean
###########################################################################################################
def get_max_rate_dict(
    allDat: Dict[str, Any],
    celltype: List[str],
    maxrate_range: Tuple[int, int] = (198, 202),
    baseline_range: Tuple[int, int] = (0, 100),
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute maximum-rate and baseline-rate dictionaries for given cell types,
    then reverse the dictionary structure.

    This function:
    1. Computes mean rates over `maxrate_range`.
    2. Computes baseline mean rates over `baseline_range`.
    3. Inserts a 'baseline' entry into the max-rate dictionary using
       baseline values from the 'ds' condition.
    4. Returns the reversed dictionary using `sbf.reverse_dict`.

    Parameters
    ----------
    allDat : Dict[str, Any]
        Nested dataset dictionary passed to `generate_mean_rate`.
        Expected structure must be compatible with that function.
    celltype : List[str]
        List of cell type names to process.
    maxrate_range : Tuple[int, int], optional
        Index range (start, end) over which to compute maximum rate.
        Default is (198, 202).
    baseline_range : Tuple[int, int], optional
        Index range (start, end) over which to compute baseline rate.
        Default is (0, 100).

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Reversed dictionary of maximum and baseline rates.
        The exact structure depends on `sbf.reverse_dict`.

    Raises
    ------
    KeyError
        If expected keys (e.g., 'ds') are missing from intermediate results.
    """
    # Compute max-rate data
    maxrate_dat = generate_mean_rate(
        allDat,
        celltype,
        xrange=maxrate_range,
    )

    # Compute baseline data
    baserate_dat = generate_mean_rate(
        allDat,
        celltype,
        xrange=baseline_range,
    )

    # Initialize baseline entry
    maxrate_dat["baseline"] = {ctype: np.array([]) for ctype in celltype}

    # Populate baseline values using the 'ds' condition
    for ctype in celltype:
        maxrate_dat["baseline"][ctype] = baserate_dat["ds"][ctype]

    return sbf.reverse_dict(maxrate_dat)


