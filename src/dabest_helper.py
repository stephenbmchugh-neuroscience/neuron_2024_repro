######## Dabest helper functions #################################
from typing import Any, Optional, Sequence, Tuple, Iterable
from matplotlib.figure import Figure
from matplotlib.axes import Axes

import numpy as np
import pandas as pd
import scipy.stats as stats
import dabest as db2
import smBaseFunctions3 as sbf
import analysis as an

import matplotlib.pyplot as plt
import seaborn as sns
from my_mpl_defaults import *
from plotting import cm2inch


def plot_dabest_swarm_contrast(
    analysis_of_long_df: Any,
    df_idx: Sequence[Any],
    show_pairs: bool = False,
    diff_type: str = 'mean',
    swlab: str = 'Firing rate (Hz)',
    size_scale: float = 2,
    fwd: float = 1.25,
    fht: float = 4,
    pad: float = 1.5,
    swarm_ylim: Optional[Sequence[float]] = (-5, 60),
    contrast_ylim: Optional[Sequence[float]] = (-5, 20),
    swarm_maj_loc: float = 20,
    contrast_maj_loc: float = 10,
    fscale: Sequence[float] = (5, 6, 7),
    raw_marker_size: float = 0.05,
    jitter: float = 3,
    my_color_palette: Optional[Iterable[Any]] = None
) -> Tuple[Figure, Axes, Axes]:
    """
    Convenience wrapper that configures parameters and calls `smdabest_plot` to
    create a dabest swarm + contrast plot.

    Parameters
    ----------
    analysis_of_long_df
        The analysis object containing grouped/long-form data and .mean_diff or
        .median_diff plotting interfaces (e.g. a dabest-like analysis object).
        The function calls `analysis_of_long_df.mean_diff.plot` or
        `analysis_of_long_df.median_diff.plot` depending on `diff_type`.
    df_idx
        Index or iterable used to compute figure width scaling. The function
        computes `len(sbf.flatten(df_idx))` in the original code; any sequence
        compatible with your `sbf.flatten` call is fine.
    show_pairs
        If True, show paired raw datapoint lines in the dabest plot.
    diff_type
        Which effect-size type to plot. Valid values: `'mean'` or `'median'`.
    swlab
        Label for the swarm (left) axis.
    size_scale
        Global scaling factor for sizes, fonts and figure dimensions.
    fwd, fht
        Base figure width and height; these are multiplied by `size_scale` and
        by the number of groups derived from `df_idx`.
    pad
        Tick padding applied to axes.
    swarm_ylim, contrast_ylim
        Y-limits for the swarm and contrast axes respectively (min, max).
    swarm_maj_loc, contrast_maj_loc
        Major-tick spacing for swarm and contrast axes.
    fscale
        Font size scale tuple (small, medium, large).
    raw_marker_size
        Marker size for raw points in the swarm plot.
    jitter
        Jitter magnitude for raw points (present in original signature but not
        used directly in this wrapper).
    my_color_palette
        Optional iterable of colors to pass to the plotting routine. If None,
        a default palette is used.

    Returns
    -------
    fig, swarm_ax, contrast_ax
        Matplotlib Figure and the swarm and contrast Axes returned from
        `smdabest_plot`.

    Notes
    -----
    - This wrapper suppresses `dabest` UserWarnings during plotting.
    - The function expects the plotting helper `smdabest_plot` and variables
      referenced in the original context (e.g. `sbf.flatten`, color constants)
      to be available in the calling module.
    """
    import warnings
    if diff_type == 'mean':
        eslab = 'Mean ' + r'$\Delta$'
    elif diff_type == 'median':
        eslab = 'Median ' + r'$\Delta$'
    #######################################################################################################
    if my_color_palette is None:
        my_color_palette = [gray2, GREEN, PURPLE, BLUE, ORNG,
                            gray2, GREEN, PURPLE, BLUE, ORNG,
                            gray2, GREEN, PURPLE, BLUE, ORNG
                           ]
    #####################################################################################################
    fwd, fht = fwd * len(sbf.flatten(df_idx)) * size_scale, fht * size_scale
    pad = pad * size_scale
    font_scale = [x * size_scale for x in fscale]
    #####################################################################################################
    bw_method = 1  # sbf.calc_mean_bw_1var(idata,innerkey)
    float_contrast = False
    ########################################################################################################
    warnings.filterwarnings("ignore", category=UserWarning, module="dabest")
    fig, swarm_ax, contrast_ax = smdabest_plot(
        analysis_of_long_df,
        float_contrast=float_contrast,
        diff_type=diff_type,
        figsize=[fwd, fht],
        size_scale=size_scale,
        raw_marker_size=raw_marker_size,
        swarm_ylim=swarm_ylim,
        contrast_ylim=contrast_ylim,
        my_color_palette=my_color_palette,
        swlab=swlab,
        eslab=eslab,
        fs=font_scale,
        show_pairs=show_pairs,
        maj_loc=[swarm_maj_loc, contrast_maj_loc],
        bw_method=bw_method
    )
    ########################################################################################################
    return fig, swarm_ax, contrast_ax


def smdabest_plot(
    analysis_of_long_df: Any,
    float_contrast: bool = False,
    diff_type: str = 'mean',
    figsize: Sequence[float] = [8, 8],
    size_scale: float = 2,
    raw_marker_size: float = 1,
    swarm_ylim: Optional[Sequence[float]] = None,
    contrast_ylim: Optional[Sequence[float]] = None,
    my_color_palette: Optional[Iterable[Any]] = None,
    swlab: str = 'Decoding accuracy',
    eslab: str = 'Median ' + r'$\Delta$',
    fs: Sequence[float] = [9, 12, 16],
    show_pairs: bool = False,
    maj_loc: Optional[Sequence[float]] = None,
    bw_method: float = 0.5,
    pad: float = 1.5,
    lw: float = 0.5
) -> Tuple[Figure, Axes, Axes]:
    """
    Create and format a dabest-style swarm + contrast plot.

    This function calls the `plot` method on `analysis_of_long_df.mean_diff` or
    `analysis_of_long_df.median_diff` (depending on `diff_type`) and then
    formats axes labels, tick locators, and basic styles.

    Parameters
    ----------
    analysis_of_long_df
        The analysis object containing grouped/long-form data and `.mean_diff`
        or `.median_diff` attributes with a `.plot` method that accepts the
        arguments used below (custom_palette, float_contrast, fig_size, etc.).
        Typically this is the result of preparing data for dabest plotting.
    float_contrast
        If True, render the contrast axis as a floating axis (dabest option).
    diff_type
        'mean' or 'median' — selects which summary/difference object to plot.
    figsize
        [width, height] in inches (or units expected by `cm2inch` in the
        original code) to pass to the plotting routine.
    size_scale
        Scaling factor applied to linewidths, marker sizes and font sizes.
    raw_marker_size
        Size for raw data markers when plotting the swarm.
    swarm_ylim, contrast_ylim
        Limits for the swarm and contrast y-axes.
    my_color_palette
        Iterable of colors used as a palette. If None, a default palette is used.
    swlab, eslab
        Labels for swarm and effect-size (contrast) axes respectively.
    fs
        Tuple/list of font sizes (small, medium, large).
    show_pairs
        If True, show paired lines between raw points in the dabest output.
    maj_loc
        Optional two-item sequence specifying major tick spacing for
        [swarm_major_locator, contrast_major_locator]. If None, sensible
        defaults are used.
    bw_method
        Bandwidth method parameter (kept for compatibility with original code).
    pad
        Tick padding applied to axes.
    lw
        Base line width (will be multiplied by `size_scale`).

    Returns
    -------
    fig, swarm_ax, contrast_ax
        Matplotlib Figure and the swarm and contrast Axes created by the
        underlying dabest plotting routine.

    Notes
    -----
    - This function depends on several module-level names from the original
      codebase (e.g. `gray2`, color constants like `RED`, `cm2inch`, `Ticker`,
      and `sbf`). Ensure those are available where this function is used.
    - The plotting backend invoked is expected to return an object `fig`
      whose `axes` list contains the swarm axis at index 0 and the contrast
      axis at index 1.
    """
    fwd, fht = figsize[0], figsize[1]

    SMALL_SIZE = fs[0]
    MEDIUM_SIZE = fs[1]
    BIGGER_SIZE = fs[2]

    if maj_loc is None:
        swarm_maj_loc = .2
        contrast_maj_loc = swarm_maj_loc / 1
    else:
        swarm_maj_loc = maj_loc[0]
        contrast_maj_loc = maj_loc[1]
        
    lw = lw * size_scale
    contrast_marker_size = 1 * size_scale
    rlcol = gray2
    rlkwargs = {'linewidth': lw, 'color': rlcol, 'linestyle': '--'}

    raw_desat = 1.0
    contrast_desat = 1.0

    if diff_type == 'mean':
        diff_type_plot = analysis_of_long_df.mean_diff.plot
    elif diff_type == 'median':
        diff_type_plot = analysis_of_long_df.median_diff.plot
    ####################################################################################################
    fig = diff_type_plot(
        custom_palette=my_color_palette,
        float_contrast=float_contrast,
        fig_size=cm2inch(fwd, fht),
        show_pairs=show_pairs,
        raw_bars=False,
        raw_marker_size=raw_marker_size,
        swarm_side="center",
        raw_desat=raw_desat,
        raw_ylim=swarm_ylim,
        contrast_bars=False,
        contrast_marker_kwargs={'markersize': contrast_marker_size},
        contrast_errorbar_kwargs={'lw': lw},
        contrast_desat=contrast_desat,
        contrast_ylim=contrast_ylim,
        contrast_paired_lines=False,
        delta_text=False,
        group_summaries_kwargs={'lw': lw},
        reflines_kwargs=rlkwargs,
    )

    swarm_ax = fig.axes[0]
    contrast_ax = fig.axes[1]

    swarm_ax.set_ylabel(swlab, fontsize=MEDIUM_SIZE)
    contrast_ax.set_ylabel(eslab, fontsize=MEDIUM_SIZE)

    swarm_ax.yaxis.set_major_locator(Ticker.MultipleLocator(swarm_maj_loc))
    contrast_ax.yaxis.set_major_locator(Ticker.MultipleLocator(contrast_maj_loc))

    for ax_ in [swarm_ax, contrast_ax]:
        ax_.xaxis.set_tick_params(labelsize=SMALL_SIZE, width=lw, length=lw * 3)
        ax_.yaxis.set_tick_params(labelsize=SMALL_SIZE, width=lw, length=lw * 3)
        for aa in ['x', 'y']:
            ax_.tick_params(axis=aa, pad=pad)

    return fig, swarm_ax, contrast_ax
#############################################################################################################
def adjust_dabest_labels(
    swarm_ax: Axes,
    contrast_ax: Axes,
    df_idx: Sequence[Any],
    max_char: int = 8,
    str_indx: int = 1,
    upper_case: bool = True
) -> Tuple[Axes, Axes]:
    """
    Adjust x-axis tick labels for swarm and contrast axes in a dabest plot.

    This function generates formatted x-axis labels from `df_idx` using
    `generate_dabest_xtlabs`, then applies them to both the swarm and
    contrast axes.

    Parameters
    ----------
    swarm_ax
        Matplotlib Axes corresponding to the swarm (raw data) plot.
    contrast_ax
        Matplotlib Axes corresponding to the contrast (effect size) plot.
    df_idx
        Sequence used to derive group labels (typically the same grouping
        structure used when constructing the dabest analysis object).
        Passed directly to `generate_dabest_xtlabs`.
    max_char
        Maximum number of characters allowed per label. Longer labels
        are truncated inside `generate_dabest_xtlabs`.
    str_indx
        Index used when extracting label components from `df_idx`
        (depends on how your grouping/index structure is defined).
    upper_case
        If True, convert labels to uppercase.

    Returns
    -------
    swarm_ax, contrast_ax
        The modified Matplotlib Axes objects with updated x-axis labels.

    Notes
    -----
    - This function depends on `generate_dabest_xtlabs` being available
      in the current namespace.
    - It assumes both axes already have the correct number of tick positions
      corresponding to the generated labels.
    """
    sw_xtlabs, con_xtlabs = generate_dabest_xtlabs(
        df_idx,
        max_char=max_char,
        str_indx=str_indx,
        upper_case=upper_case
    )

    swarm_ax.set_xticklabels(sw_xtlabs)
    contrast_ax.set_xticklabels(con_xtlabs)

    return swarm_ax, contrast_ax
########################################################################
def generate_dabest_analysis(idata,
                             outerkey,
                             innerkey,
                             col_head,
                             col_groups,
                             df_idx,
                             paired='baseline',
                             x='Cell_Event',
                             y='Data',
                             id_col='id'):
    
    idata = an.nan_free_dict(idata,outerkey,innerkey)
    df = dabest_long_df_2var(idata,innerkey,outerkey,col_groups,col_head)
    #####################################################################
    analysis_of_long_df = db2.load(df,
                                  x=x,
                                  y=y,
                                  idx=((df_idx)),
                                  paired=paired,
                                  id_col=id_col)

    return analysis_of_long_df
######################################################################
def dabest_long_df_1var(idata,key,col_groups,col_head):
    '''

    '''
    data_r = np.array([])
    ilab_r = np.array([])
    id_r = np.array([],dtype=int)
    group_r = np.array([])
    
    for iind, ikey in enumerate(key):
        data_r = np.r_[data_r,idata[ikey]]
        ilab_r = np.r_[ilab_r,np.repeat(col_groups[ikey],len(idata[ikey]))]
        id_r = np.r_[id_r, np.arange(0,len(idata[ikey]))]
        glab = col_groups[ikey]
        group_r = np.r_[group_r,np.repeat(glab,len(idata[ikey]))]
            
    df = pd.DataFrame({col_head[0]: id_r,
                       col_head[1]: ilab_r,
                       col_head[2]: data_r,
                       col_head[3]: group_r
                       })
    return df
######################################################################
def dabest_long_df_2var(idata,innerkey,outerkey,col_groups,col_head):
    '''

    '''
    data_r = np.array([])
    ilab_r = np.array([])
    olab_r = np.array([])
    id_r = np.array([],dtype=int)
    group_r = np.array([])
    
    for iind, ikey in enumerate(innerkey):
        for oind, okey in enumerate(outerkey):
            data_r = np.r_[data_r,idata[ikey][okey]]
            ilab_r = np.r_[ilab_r,np.repeat(col_groups[ikey],len(idata[ikey][okey]))]
            olab_r = np.r_[olab_r,np.repeat(okey,len(idata[ikey][okey]))]
            id_r = np.r_[id_r, np.arange(0,len(idata[ikey][okey]))]
            glab = col_groups[ikey] + '_' + okey
            group_r = np.r_[group_r,np.repeat(glab,len(idata[ikey][okey]))]

    df = pd.DataFrame({col_head[0]: id_r,
                       col_head[1]: ilab_r,
                       col_head[2]: olab_r,
                       col_head[3]: data_r,
                       col_head[4]: group_r
                       })
    return df 
###########################################################################################################
def generate_dabest_xtlabs(df_idx,max_char=4,str_indx=0,upper_case=True):
    '''
    
    '''
    sw_xtlabs = []
    con_xtlabs = []
    
    for each_ in df_idx:
        n_groups = len(each_)
        if upper_case:
            each_list = [x.split('_',3)[str_indx][:max_char].upper() for x in each_]
        else:
            each_list = [x.split('_',3)[str_indx][:max_char] for x in each_]

        sw_xtlabs.append(each_list)
        ####################################################################
        ostr = ['']
        for group in range(1,n_groups):
            ostr.append("\nminus\n".join([each_list[group],
                                          each_list[0]])
                       )
        con_xtlabs.append(ostr)
        ####################################################################  
    return sbf.flatten(sw_xtlabs),sbf.flatten(con_xtlabs)
############################################################################################################
def update_xticklabels(sw_xtlabs,con_xtlabs,swarm_ax,contrast_ax):

    for ax_,tl in zip([swarm_ax,contrast_ax],[sw_xtlabs,con_xtlabs]):
        ax_.set_xticklabels(tl)
#############################################################################################################
