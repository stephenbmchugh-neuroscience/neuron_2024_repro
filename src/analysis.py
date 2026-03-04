"""analysis.py
Statistical analysis / model-fitting functions.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
import smBaseFunctions3 as sbf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import copy
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

################################################################################################################
def generate_mean_rate(allDat,celltype,xrange=(0,150)):
    '''
    
    '''
    odata = {}
    for ikey,val in allDat.items():
        tempdict = {}
        for cindx,ctype in enumerate(celltype):
            tempdat = np.squeeze(val[ctype][0])
            slice_data = tempdat[xrange[0]:xrange[1]]
            tempdict[ctype] = np.nanmean(tempdat[xrange[0]:xrange[1]],axis=0)
        odata[ikey] = tempdict
        
    return odata
####################################################################################
def bin_array(idata, axis=0, binstep=10, binsize=10, func=np.nanmean):
    '''

    '''
    idata = np.array(idata)
    dims = np.array(idata.shape)
    argdims = np.arange(idata.ndim)
    argdims[0], argdims[axis]= argdims[axis], argdims[0]
    idata = idata.transpose(argdims)
    idata = [func(np.take(idata,np.arange(int(i*binstep),int(i*binstep+binsize)),0),0) 
             for i in np.arange(dims[axis]//binstep)]
    
    odata = np.array(idata).transpose(argdims)
    
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
def get_max_rate_dict(allDat: Dict,
                      celltype: List,
                      maxrate_range=(198,202),
                      baseline_range=(0,100)):
    '''

    '''
    maxrate_dat = generate_mean_rate(allDat,
                                     celltype,
                                     xrange=maxrate_range
                                     )
    baserate_dat = generate_mean_rate(allDat,
                                     celltype,
                                     xrange=(0,100)
                                     )
    ###############################################################################################
    maxrate_dat['baseline'] = {'pdg':np.array([]),
                               'p3':np.array([]),
                               'p1':np.array([])}
    ###############################################################################################
    for cindx,ctype in enumerate(celltype):
        maxrate_dat['baseline'][ctype] = baserate_dat['ds'][ctype]
    
    return sbf.reverse_dict(maxrate_dat)


def train_simple_model(X: np.ndarray, y: np.ndarray, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state, test_size=0.2)
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return clf, score
