"""analysis.py
Data reduction, z-scoring and statistical analysis
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import rankdata
import smBaseFunctions3 as sbf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA,FastICA

import copy
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any, Callable, Mapping, Union

import smBaseFunctions3 as sbf
import smCofiringFunctions as scf

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

def gini(
    array: Union[Iterable[float], np.ndarray],
    constant: float = 0.0,
    use_rankdata: bool = False,
) -> float:
    """
    Calculate the Gini coefficient for a 1-D numeric array.

    The Gini coefficient is a measure of statistical dispersion intended to
    represent the inequality in a distribution (0 = perfect equality,
    1 = maximal inequality for non-negative values).

    Parameters
    ----------
    array : Iterable[float] | np.ndarray
        Input data. Will be flattened to 1D and converted to float.
    constant : float, optional
        A small constant added to every element to avoid zero division or to
        shift values (useful if there are zeros). Default is 0.0.
        Note: adding a constant changes the interpretation of the Gini.
    use_rankdata : bool, optional
        If True, use `scipy.stats.rankdata` to compute ranks which deals with
        ties by averaging; if False (default) uses consecutive integer ranks
        after sorting. `rankdata` is slightly more correct for tied values
        but slightly slower.

    Returns
    -------
    float
        The Gini coefficient in [0, 1] for non-negative input arrays.

    Raises
    ------
    ValueError
        If `array` is empty or the sum of values is zero (after shifting),
        which would make the coefficient undefined.
    """
    # Convert to numpy array, flatten and cast
    arr = np.asarray(array, dtype=float).flatten()

    if arr.size == 0:
        raise ValueError("Input array is empty; Gini is undefined.")

    # Replace NaNs and infs: fail early (could also choose to ignore NaNs)
    if np.isnan(arr).any() or np.isinf(arr).any():
        raise ValueError("Input contains NaN or infinite values; clean the data first.")

    # Shift negative values to make all non-negative
    min_val = np.min(arr)
    if min_val < 0:
        arr = arr - min_val

    # Add small constant if requested (default 0)
    if constant != 0.0:
        arr = arr + float(constant)

    total = np.sum(arr)
    if total == 0:
        raise ValueError("Sum of the (shifted) array is zero; Gini is undefined.")

    # Sort values (non-decreasing)
    arr_sorted = np.sort(arr)

    # Compute ranks
    if use_rankdata:
        ranks = rankdata(arr_sorted)  # average ranks for ties
    else:
        # simple 1..n ranks (works fine even with ties if you don't need averaged ranks)
        n = arr_sorted.size
        ranks = np.arange(1, n + 1, dtype=float)

    n = arr_sorted.size
    # Gini formula (numerically stable form)
    gini_coeff = (np.sum((2.0 * ranks - n - 1.0) * arr_sorted)) / (n * total)

    # Guard numerical noise: clip to [0,1]
    return float(np.clip(gini_coeff, 0.0, 1.0))

def generate_mean_sparsity_from_IFR(IFR_data, lcond_list, mouseID):
    """
    Compute mean sparsity values from instantaneous firing rate (IFR) data.

    For each condition in `lcond_list` and each day in `IFR_data`,
    this function computes a sparsity metric (default: Gini coefficient)
    across trials, then returns the mean sparsity per day per condition.

    Parameters
    ----------
    IFR_data : list
        List of length n_days. Each element corresponds to one day and
        must be a dictionary indexed by condition names in `lcond_list`.
        Each entry should be a 2D NumPy array of shape (n_timepoints, n_trials)
        representing instantaneous firing rate data.

    lcond_list : list
        List of condition labels (keys) used to index into each day's
        IFR_data dictionary.

    mouseID : list or array-like
        Identifier(s) for each day. Currently not used internally,
        but included for compatibility or future extensions.

    Returns
    -------
    odata : dict
        Dictionary where:
            keys   = condition labels from `lcond_list`
            values = NumPy array of shape (n_days,)
                     containing mean sparsity values per day
                     (NaNs ignored in averaging).

    Notes
    -----
    - Trials with zero total activity (sum == 0) are assigned NaN.
    - Mean sparsity per day is computed using `np.nanmean`.
    - NaNs propagate if all trials for a day/condition are invalid.
    """
    n_days = len(IFR_data)
    odata = {}

    for lindx, lcond in enumerate(lcond_list):
        mean_sparsity = []

        for dindx in np.arange(n_days):
            idata = IFR_data[dindx][lcond]
            temp_data = []

            for trial in np.arange(idata.shape[1]):
                if np.nansum(idata[:, trial]) > 0:
                    temp_data.append(gini(idata[:, trial]))
                else:
                    temp_data.append(np.nan)

            mean_sparsity.append(np.nanmean(temp_data))

        odata[lcond] = np.array(mean_sparsity)

    return odata

def create_bool_dict(
    iDict: Dict[str, List[Dict[str, np.ndarray]]],
    bsnm_list: List[Any]
) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """
    Convert all arrays inside a nested event → mouse → condition dictionary to boolean.

    Parameters
    ----------
    iDict : dict
        Nested dictionary structured as:
            {
                event_name: [
                    {condition_name: np.ndarray (n_cells, n_events), ...},  # mouse 0
                    {condition_name: np.ndarray (n_cells, n_events), ...},  # mouse 1
                    ...
                ],
                ...
            }

        Each array is assumed to contain numeric values that can be cast to boolean.

    bsnm_list : list
        List of mouse identifiers. Used to ensure consistent indexing
        across mice (length should match the number of mouse entries per event).

    Returns
    -------
    dict
        Dictionary with the same structure as `iDict`, but with all arrays
        converted to boolean type (dtype=bool).

    Notes
    -----
    - The original `iDict` is not modified.
    - Conversion is performed using `array.astype(bool)`:
        - Zero values become False
        - Non-zero values become True
    """

    bool_IFR_dict: Dict[str, List[Dict[str, np.ndarray]]] = {}

    for event_key, event_val in iDict.items():
        temp_list: List[Dict[str, np.ndarray]] = []

        for mindx, _ in enumerate(bsnm_list):
            temp_dict: Dict[str, np.ndarray] = {}

            for cond_key, cond_array in event_val[mindx].items():
                temp_dict[cond_key] = cond_array.astype(bool)

            temp_list.append(temp_dict)

        bool_IFR_dict[event_key] = temp_list

    return bool_IFR_dict
    
def select_random_subset(arr,max_bin,axis=1,nprs=100):
    '''
    number_of_rows = arr.shape[0]
    number_of_cols = arr.shape[1]
    '''
    np.random.seed(nprs)
    
    number_of_ = arr.shape[axis]
    random_indices = np.random.choice(number_of_, size=max_bin, replace=True)
    
    if axis == 1:
        return arr[:,random_indices]
    else:
        return arr[random_indices,:]
        

def get_max_bin(event):
    max_bin_dict = {'ds':104, 'swr': 106}
    return max_bin_dict[event]

def generate_pv_similarity_data(
    iDict: Mapping[str, Sequence[Any]],
    new_mouseID: Sequence[Any],
    event_pair_list: Sequence[Tuple[str, str]],
    rand_subset: bool = True,
    binarize: bool = True,
    shuffle_list: Sequence[bool] = (False, True),
    event_list: Sequence[str] = ("ds", "swr"),
    output_dat: str = "mean",
    cond: str = "pulse",
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute pairwise population-vector (PV) similarity summaries for many mice
    and event-pairs, optionally with shuffled data.

    This function iterates over `shuffle_list` and `event_pair_list`, for each
    mouse in `new_mouseID`, extracts the event-specific data from `iDict`,
    optionally performs random subset selection and binarization, computes a
    similarity matrix via `corr_metric`, extracts the upper-triangle values and
    summarizes them (mean or std) across the upper-triangle for each mouse.

    Parameters
    ----------
    iDict : Mapping[str, Sequence[Any]]
        Nested data structure with at least these access semantics:
          A = iDict[event_name][mouse_index][cond]
        where A is an array-like of shape (n_units, n_bins) or similar.
        *Example*: iDict['ds'][0]['pulse'] -> numpy array for mouse 0, event 'ds'.
    new_mouseID : Sequence[Any]
        Iterable of mouse identifiers/indices. The function assumes `mindx`
        is the integer index into the second-level sequence (used as index).
    event_pair_list : Sequence[Tuple[str, str]]
        Iterable of (event1, event2) pairs to compare, e.g. [('ds','swr'), ...].
    rand_subset : bool, optional
        If True, select a random subset of bins (via `select_random_subset`)
        limited by the `max_bin` returned from `find_max_bin_1d`.
    binarize : bool, optional
        If True, cast A and B arrays to boolean before computing the metric.
    shuffle_list : Sequence[bool], optional
        Sequence indicating whether to use the original data or a shuffled
        variant of B. Defaults to (False, True).
    event_list : Sequence[str], optional
        List of event names used when determining the `max_bin`.
    output_dat : {'mean', 'std'} or str, optional
        If 'mean' (default) the function appends the mean of the upper-triangle
        similarity values; otherwise it appends the standard deviation.
    cond : str, optional
        Condition key used when indexing into the per-mouse dicts, e.g. 'pulse'.

    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Outer dict keyed by either 'data' or 'shuffle' (depending on shuffle_list
        values). Each value is a dictionary mapping "<event1>_<event2>" to a
        1-D numpy array of summary values (one per mouse in new_mouseID).

    Raises
    ------
    KeyError
        If expected keys are missing from `iDict` (e.g. event names or `cond`).
    ValueError
        If `output_dat` is not recognized (neither 'mean' nor 'std').
    """
    all_sim_dict: Dict[str, Dict[str, np.ndarray]] = {}

    for shuff_indx, shuffle in enumerate(shuffle_list):
        sim_dat: Dict[str, np.ndarray] = {}
        for ev_indx, event_pair in enumerate(event_pair_list):
            temp_dat: List[float] = []

            for mindx, bsnm in enumerate(new_mouseID):
                event_1, event_2 = event_pair

                # determine how many bins are usable for this mouse & condition
                max_bin = find_max_bin_1d(iDict, mindx, event_list=event_list, cond=cond)

                A = iDict[event_1][mindx][cond]
                B = iDict[event_2][mindx][cond]

                if rand_subset:
                    # assume select_random_subset returns an array shaped like A/B
                    A = select_random_subset(A, max_bin, axis=1)
                    B = select_random_subset(B, max_bin, axis=1)

                if binarize:
                    A = A.astype(bool)
                    B = B.astype(bool)

                if shuffle:
                    rng = np.random.default_rng()
                    B = rng.permuted(B, axis=0)
                    shuff_key = "shuffle"
                else:
                    shuff_key = "data"

                # compute similarity matrix between columns (units x bins -> transpose)
                sim_mat = corr_metric(A.T, B.T)

                # safely extract upper-triangle; handle small/malformed matrices
                try:
                    upper_tri, _ = scf.gen_upper_tri(sim_mat)
                except Exception:
                    # keep consistent shape when extraction fails
                    upper_tri = np.array([])

                # compute requested summary statistic
                if upper_tri.size == 0:
                    mean_utc = np.nan
                else:
                    mean_utc = np.nanmean(upper_tri)

                if output_dat == "mean":
                    temp_dat.append(mean_utc)
                elif output_dat == "std":
                    temp_dat.append(np.nanstd(upper_tri))
                else:
                    raise ValueError("output_dat must be 'mean' or 'std'")

            okey = f"{event_1}_{event_2}"
            sim_dat[okey] = np.array(temp_dat)

        all_sim_dict[shuff_key] = sim_dat

    return all_sim_dict


def get_single_similarity_matrix(
    iDict: Mapping[str, Sequence[Any]],
    mindx: int,
    event_1: str,
    event_2: str,
    cond: str = "pulse",
    rand_subset: bool = True,
    nprs: int = 10,
    max_bin: int = 101,
    binarize: bool = True,
) -> np.ndarray:
    """
    Return a single similarity matrix between two events for one mouse / condition.

    Parameters
    ----------
    iDict : Mapping[str, Sequence[Any]]
        See `generate_pv_similarity_data` for expected structure.
    mindx : int
        Integer index of the mouse to access in the per-event sequence.
    event_1 : str
        Name of the first event (source for A).
    event_2 : str
        Name of the second event (source for B).
    cond : str, optional
        Condition key used when indexing into the per-mouse dict.
    rand_subset : bool, optional
        If True, use `select_random_subset` to subsample bins; passes `nprs`
        as the number of random draws.
    nprs : int, optional
        Number of random subset repetitions passed to `select_random_subset`.
    max_bin : int, optional
        Maximum number of bins used for subset selection (passed to selector).
    binarize : bool, optional
        If True, cast arrays to boolean before computing the similarity metric.

    Returns
    -------
    np.ndarray
        Similarity matrix (2-D)`.

    Raises
    ------
    KeyError
        If requested event/condition/mouse index is not present in `iDict`.
    """
    A = iDict[event_1][mindx][cond]
    B = iDict[event_2][mindx][cond]

    if rand_subset:
        A = select_random_subset(A, max_bin, axis=1, nprs=nprs)
        B = select_random_subset(B, max_bin, axis=1, nprs=nprs)

    if binarize:
        A = A.astype(bool)
        B = B.astype(bool)

    sim_mat = corr_metric(A.T, B.T)

    return sim_mat

def corr_metric(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:,None]
    B_mB = B - B.mean(1)[:,None]
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);
    # Finally get corr
    dist = np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
    dist[np.isnan(dist)] = 0
    return dist

def cosine_metric(A,B):
    # No Row-wise mean is needed for the cosine similarity, sensible to shifts/bias
    A_mA = A 
    B_mB = B
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);
    # Finally get corr
    dist = np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
    dist[np.isnan(dist)] = 0
    return dist

def spearman_metric(A,B,axis=0):
    """spearman rho as correlation metric
    """
    A[np.isnan(A)] = 0 # convert nans to 0s
    B[np.isnan(B)] = 0
    rho,_ = stats.spearmanr(A,B,axis=0)
    return rho[:A.shape[1], B.shape[1]:]
       
def find_max_bin_1d(iDict,mindx,event_list,cond='pulse',axis=1):
    '''
    Helper function to provide balanced input to training models
    or for comparing population similarity etc.
    '''
    max_bin = 10e20
    for ev_indx,event in enumerate(event_list):
        idata = iDict[event][mindx][cond].shape[axis]
        max_bin = np.nanmin([max_bin,idata])        
    return int(max_bin)

def generate_pca(iDict,bsnm_list,event_list=['ds','ds1','ds2','swr'],cond='pulse',thrs=0.9,norm=True,min_comp=2):
    ''' 
    wrapper for calc_pca_single_matrix
    '''
    pc_dict = {}
    for key,val in iDict.items():
        tempdat = []
        for mindx,mouse in enumerate(bsnm_list):
            max_bin = find_max_bin_1d(iDict,
                                      mindx,
                                      event_list,
                                      cond=cond)
            idata = val[mindx][cond]
            idata = select_random_subset(idata,max_bin=max_bin,axis=1)
            pc_var = calc_pca_single_matrix(idata,thrs=thrs,norm=norm,min_comp=min_comp)
            tempdat.append(pc_var)
        pc_dict[key] = np.array(tempdat)
    return pc_dict

def calc_pca_single_matrix(
    idata: np.ndarray,
    thrs: float = 0.8,
    norm: bool = False,
    min_comp: int = 5,
) -> Union[int, float]:
    """
    Compute the number of principal components required to exceed a
    cumulative explained variance threshold for a single data matrix.

    Parameters
    ----------
    idata : np.ndarray
        2D input array of shape (n_components, n_bins) or (n_samples, n_features).
        PCA is performed along axis 0 using all rows as components.
    thrs : float, optional
        Cumulative explained variance threshold (between 0 and 1).
        The function returns the smallest component index for which the
        cumulative explained variance exceeds this threshold.
        Default is 0.8 (80% variance explained).
    norm : bool, optional
        If True, normalize the resulting component index by dividing by
        the total number of components. Default is False.
    min_comp : int, optional
        Minimum number of components required to compute a valid result.
        If the number of components in `idata` is less than this value,
        the function returns np.nan. Default is 5.

    Returns
    -------
    int | float
        Index (0-based) of the first principal component at which the
        cumulative explained variance exceeds `thrs`.

        Returns:
        - np.nan if the threshold is never reached.
        - np.nan if the number of components is less than `min_comp`.
        - A float if `norm=True` (normalized index).

    Notes
    -----
    - The returned value is a 0-based index.
    - If normalization is enabled, the output is scaled by the total
      number of components.
    - PCA is fit using `sklearn.decomposition.PCA`.
    """
    if idata.ndim != 2:
        raise ValueError("idata must be a 2D array.")

    nComp = idata.shape[0]

    if nComp < min_comp:
        return np.nan

    pca = PCA(n_components=nComp)
    odata = pca.fit(idata)
    eval_ = odata.explained_variance_ratio_

    cumsum_eval = np.cumsum(eval_)

    try:
        pcvar_out: Union[int, float] = int(np.where(cumsum_eval > thrs)[0][0])
    except IndexError:
        return np.nan

    if norm:
        pcvar_out = pcvar_out / nComp

    return pcvar_out

