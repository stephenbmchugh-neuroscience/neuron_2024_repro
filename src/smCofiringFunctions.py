######################################################################################
## Import libraries
######################################################################################
import pandas as pd
import numpy as np
import sys as sys
import math as math
import os as os
import time as time
import vBaseFunctions3 as vbf
import matplotlib.pyplot as plt
import matplotlib.ticker as Ticker
import scipy as scipy
import scipy.stats as stats
import scipy.ndimage as spim
import scipy.signal as sig
import fnmatch
import smBaseFunctions3 as sbf

from math import sqrt
from sklearn.utils.extmath import squared_norm
######################################################################################
def get_cell_inds(units,mouse,ctype):
    '''
    
    '''
    temp_cells = units[mouse]['des'].values
    cell_inds = [index for index,value in enumerate(temp_cells) if value==ctype] # need to double check this!!!
    orig_inds = [x+2 for x in cell_inds]

    return cell_inds,orig_inds
######################################################################################
def get_all_cell_inds(mouseID,units,celltype):
    '''
    
    '''
    allMiceInds = []
    allMiceOrigInds = []
    for mindx,mouse in enumerate(mouseID):
        cell_inds = {}
        orig_inds = {}
        for cindx,ctype in enumerate(celltype):
            cell_inds[ctype],orig_inds[ctype] = get_cell_inds(units,mindx,ctype)
        allMiceInds.append(cell_inds)
        allMiceOrigInds.append(orig_inds)

    return allMiceInds,allMiceOrigInds
######################################################################################
def generate_IFR_celltypes(FRList,celltype,sessions,units,mouse):
    '''
    This will generate an IFR for one mouse for all sessions
    You can pass either a single celltype (in square brackets) or a list
    and it will concatenate different cell types 
    '''
    IFR = {}
    for sindx,sess in enumerate(sessions):
        timeXtrials = int(FRList[sess][mouse].shape[0] * FRList[sess][mouse].shape[2])
        tempDat = np.empty([0,timeXtrials])
        for cindx,ctype in enumerate(celltype):
            cell_inds,_ = get_cell_inds(units,mouse,ctype)
            array_to_add = reshape_cellsXTrialsTime(FRList[sess][mouse][:,cell_inds,:])
            tempDat = np.concatenate((tempDat,array_to_add))
        IFR[sess] = tempDat
    odata = IFR

    return odata
##################################################################################
def generate_reshape_IFR(FRList,ctype,sessions,units,mouse):
    '''
    This will generate an IFR for one mouse for all sessions
    from a 3D (time x cells x trials) to 2D (cells x (trials x time))
    '''
    odata = {}
    cell_inds,_ = get_cell_inds(units,mouse,ctype)
    for sindx,sess in enumerate(sessions):
        odata[sess] = reshape_cellsXTrialsTime(FRList[sess][mouse][:,cell_inds,:])

    return odata
###################################################################################
def generate_smooth_reshape_IFR(FRList,ctype,sessions,units,mouse,GaussianStd=5,GaussianNPoints=0):
    '''
    Gaussian smooth 3D (time x cells x trials) then reshape to 2D (cells x (Trials x Time))
    '''
    odata = {}
    cell_inds,_ = get_cell_inds(units,mouse,ctype)
    for sindx,sess in enumerate(sessions):
        tempDat = smooth_IFR(FRList[sess][mouse][:,cell_inds,:],GaussianStd,GaussianNPoints) # yields a timebins x cells x trials matrix
        odata[sess] = reshape_cellsXTrialsTime(tempDat) # yields a cells x (trials x Time) matrix
    
    return odata
###################################################################################################
def generate_smooth_reshape_IFR_one_sess(FRList,sess,cell_inds,mouse,GaussianStd=5,GaussianNPoints=0):
    '''
    Gaussian smooth 3D (time x cells x trials) then reshape to 2D (cells x (Trials x Time))
    '''
    tempDat = smooth_IFR(FRList[sess][mouse][:,cell_inds,:],GaussianStd,GaussianNPoints) # yields a timebins x cells x trials matrix
    odata = reshape_cellsXTrialsTime(tempDat) # yields a cells x (trials x Time) matrix
    
    return odata
###################################################################################################
def smooth_IFR(iMat,GaussianStd=5,GaussianNPoints=0):
    '''
    Based on Vitor's MatrixGaussianSmooth
    But adapted to work on 3D matrix with structure: time x cells x trials 
    '''
    NormOperator=np.sum
    if GaussianNPoints < GaussianStd:
        GaussianNPoints = int(4*GaussianStd)

    GaussianKernel = sig.get_window(('gaussian',GaussianStd),GaussianNPoints)
    GaussianKernel = GaussianKernel/NormOperator(GaussianKernel)
    
    oMat = np.ones(np.shape(iMat)) * np.nan
    
    for cell_indx in range(iMat.shape[1]):
        for trial_indx in range(iMat.shape[2]):
            array_to_convolve = np.ravel(iMat[:,cell_indx,trial_indx])
            oMat[:,cell_indx,trial_indx] =  np.convolve(array_to_convolve,GaussianKernel,'same')

    return oMat
########################################################################################################
def smooth_1d(idata,GaussianStd=5,GaussianNPoints=0):
    '''
    Based on Vitor's MatrixGaussianSmooth
    '''
    import scipy.signal as sig
    
    NormOperator=np.sum
    if GaussianNPoints < GaussianStd:
        GaussianNPoints = int(4*GaussianStd)

    GaussianKernel = sig.get_window(('gaussian',GaussianStd),GaussianNPoints)
    GaussianKernel = GaussianKernel/NormOperator(GaussianKernel)
    
    array_to_convolve = np.ravel(idata)
    odata = np.convolve(array_to_convolve,GaussianKernel,'same')

    return odata
####################################################################################
def reshape_cellsXTrialsTime(idata):
    '''
    this function will reshape a 3d time x cells x trials array
    into a 2d cells x (Trials x Time) array
    ''' 
    temp_rsh = np.transpose(idata,(1,2,0))
    cells = temp_rsh.shape[0]
    trials = temp_rsh.shape[1]
    timen = temp_rsh.shape[2]
    odata = np.reshape(temp_rsh,(cells,trials*timen))

    return odata
##########################################################################
def generate_cofiring_mat(idata,ctype,sessions,simType='pear'):
    '''
    takes a dictionary of IFR matrices (cells x timebins), 
    a celltype, and list of sessions...
    and returns the correlation matrix
    '''
    odata = {}
    
    for sindx,sess in enumerate(sessions):
        tempDat = idata[ctype][sess]
        odata[sess] = calc_cofiring_matrix(tempDat,simType=simType)
    
    return odata
##########################################################################
def calc_cofiring_matrix(idata,simType='pear'):
    '''
    Inputs:
        idata is a cells x timebins numpy array
        simType is an option to specify the type of similarity to compute
    Outputs:
        odata is a matrix (len(sessions x len(sessions)
        the diagonals of odata = 0
    '''
    ######################################################################
    nCells = idata.shape[0]
    odata = np.empty([nCells,nCells])
    ######################################################################
    for ii in range(0,nCells):
        for jj in range(0,nCells):
            if ii == jj:
                odata[ii,jj] = 0 # set diagonal to zero
            else:
                vect1 = idata[ii]
                vect2 = idata[jj]
                #print vect1.shape,vect2.shape
                if simType == 'cos':
                    odata[ii,jj] = sbf.cosine_similarity(vect1,vect2)
                elif simType == 'pear':
                    odata[ii,jj],pval = stats.mstats.pearsonr(vect1,vect2)
                elif simType == 'spear':
                    odata[ii,jj],pval = stats.mstats.spearmanr(vect1,vect2)
    ######################################################################
    return odata
######################################################################################################
def cross_region_corr_matrix(allIFR,refCell,mouse,sess,refctype='p3',targctype='p1',simType='pear'):
    '''
    
    '''
    #######################################################################
    odata = []
    #######################################################################
    refIFRMat = allIFR[mouse][refctype][sess][refCell,:].reshape(1,-1)
    targIFRMat = allIFR[mouse][targctype][sess]
    for ntarg in range(targIFRMat.shape[0]):
        vect1 = np.squeeze(refIFRMat)
        vect2 = np.squeeze(targIFRMat[ntarg,:])
        if simType == 'cos':
            odata.append(sbf.cosine_similarity(vect1,vect2))
        elif simType == 'pear':
            odata.append(stats.mstats.pearsonr(vect1,vect2)[0])
        elif simType == 'spear':
            odata.append(stats.mstats.spearmanr(vect1,vect2)[0])
    return odata
###########################################################################
def gen_upper_tri(imat):
    '''
    returns upper triangle of correlation matrix (or any matrix)
    also returns the mean of this upper triangle
    '''
    iu1 = np.triu_indices(imat.shape[0],k=1)
    omat = imat[iu1]

    return omat,np.nanmean(omat[np.isfinite(omat)])
##########################################################################
def corr_metric(A,B):
    '''

    '''
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
###########################################################################
def cosine_metric(A,B):
    '''

    '''
    # No Row-wise mean inneeded for the cosine similarity, sensible to shifts/bias
    A_mA = A 
    B_mB = B
    # Sum of squares across rows
    ssA = (A_mA**2).sum(1);
    ssB = (B_mB**2).sum(1);
    # Finally get corr
    dist = np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA[:,None],ssB[None]))
    dist[np.isnan(dist)] = 0

    return dist
######################################################################################
def corrGraph(dat,THR=0):
    '''

    '''
    ccorr = np.zeros((dat.shape[1],dat.shape[1]))
    ccorr = corr_metric(dat.T,dat.T)
    for i in range(ccorr.shape[0]):
        ccorr[i,i] = 0
    cormat = ccorr.copy()
    cormat[np.abs(cormat)<THR] = 0

    return cormat
#######################################################################################
def adjust_sessions(sessions):
    '''
    
    '''
    odata = []
    sessions = [x.replace(" ","") for x in sessions]
    try: sessions.remove('f1a2')
    except: pass
    for sindx,sess in enumerate(sessions):
        if 'stim' in sess: sess = sess[:2]
        odata.append(sess)
    
    return odata
#####################################################################################################
'''
def generate_mean_corr(iMat,ctype,days_to_use,sessions,pprint=False):
    
    
    
    odata = {}
    for sindx,sess in enumerate(sessions):
        tempDat = []
        for mindx,mouse in enumerate(days_to_use):
            utc,mean_corr = gen_upper_tri(iMat[mouse][ctype][sess])
            tempDat.extend([round(x,4) for x in utc])
        odata[sess] = tempDat
        if pprint:
            print('{0:} {1:} {2:.3f} {3:.3f}'.format \
                  (sess,len(tempDat),np.nanmean(tempDat),stdErr(tempDat)))
    return odata
'''
####################################################################################################
def generate_mean_corr(iMat,ctype,days_to_use,sessions,ccMat=True,pprint=False):
    '''
    
    '''
    odata = {}
    for sindx,sess in enumerate(sessions):
        tempDat = []
        for mindx,mouse in enumerate(days_to_use):
            if ccMat:
                utc,mean_corr = gen_upper_tri(iMat[mouse][ctype][sess])
                tempDat.extend([round(x,4) for x in utc])
            else:
                tempDat.extend(iMat[mouse][ctype][sess])
        odata[sess] = tempDat
        if pprint:
            print('{0:} {1:} {2:.3f} {3:.3f}'.format(sess,len(tempDat),np.nanmean(tempDat),stdErr(tempDat)))

    return odata
####################################################################################################
def stdErr(idata):
    return np.nanstd(idata)/np.sqrt(len(idata))
####################################################################################################
def generate_merged_corr(dict1,dict2,combine=False):
    '''
    This merges two dictionaries but most is hard-coded so not for general use
    '''
    odata = {}
    # first merge the common trials
    for (k1,v1),(k2,v2) in zip (dict1.items(), dict2.items()):
        if k1[1] == '1': 
            print(k1,k2)
            odata[k1] = dict1[k1] + dict2[k1] # merge f1 and n1 trials

    # combine across all stim or nonstim
    if combine:
        #odata['stim1'] = dict1['n2'] + dict2['n3'] + dict1['n4'] + dict2['n5']
        odata['stim1'] = mean_two_lists(dict1,['n2','n4']) + mean_two_lists(dict2,['n3','n5'])
        #odata['nonstim1'] = dict1['n3'] + dict2['n2'] + dict1['n5'] + dict2['n4']
        odata['nonstim1'] = mean_two_lists(dict1,['n3','n5']) + mean_two_lists(dict2,['n2','n4'])
    # ...or keep separate trials
    else: 
        # merge stim trials    
        odata['stim1'] = dict1['n2'] + dict2['n3']
        odata['stim2'] = dict1['n4'] + dict2['n5']
        # merge nonstim trials    
        odata['nonstim1'] = dict1['n3'] + dict2['n2']
        odata['nonstim2'] = dict1['n5'] + dict2['n4']

    return odata
#######################################################################################
def mean_two_lists(idata,keys):
    
    return [np.nanmean(k) for k in zip(idata[keys[0]],idata[keys[1]])]
#######################################################################################
def merge_all_ctype(dataMetric,keys,celltype):
    '''
    
    '''
    okey = keys[0] + '_' + keys[1]

    odata = {}
    for cindx,ctype in enumerate(celltype):
        tempDat = {}
        idata = dataMetric[ctype]
        tempDat[okey]= mean_two_lists(idata,keys)
        odata[ctype] = tempDat

    return odata
#######################################################################################
def get_raster_params():
    '''

    '''
    iparams = {}
    iparams['figsize'] = [8,3]
    iparams['binwidth'] = 25
    iparams['tick_width'] = 10
    iparams['area'] = iparams['figsize'][0] / 20  # 0 to 15 point radii
    #iparams['fill_between']= False
    iparams['xlab'] = 'Cell #'
    iparams['ylab'] = 'Time (s)'

    return iparams
########################################################################################
def plot_raster(idata,iparams=None,transpose=True):
    
    '''
    '''
    if iparams==None:
        iparams = get_raster_params()
    raster = np.squeeze(idata)
    #print(raster.shape)
    if transpose:
        tempRast = (np.array(raster > 0,dtype=int)).T
    else:
        tempRast = (np.array(raster > 0,dtype=int))
    #print(tempRast.shape)
    maxTrial=tempRast.shape[0]
    #print(maxTrial)
    rastIndx = np.nonzero(tempRast[:maxTrial,:])
    ###########################################################################################
    rr,cc = 1,1
    fwid,fht = iparams['figsize'][0],iparams['figsize'][1] #8,3
    fig, ax = plt.subplots(rr,cc,figsize=sbf.cm2inch(fwid,fht))
    ############################################################################################
    ax.scatter(rastIndx[1],rastIndx[0],s=iparams['area'], c='k', alpha=0.5)
    
    ax.yaxis.set_major_locator(Ticker.MultipleLocator(iparams['tick_width']))
    xtlabs = ax.get_xticks().tolist()
    xtlabs = [format((x*iparams['binwidth'])/1000,'.0f') for x in xtlabs]
    ax.xaxis.set_ticklabels(xtlabs, fontsize=8)
    
    ytlabs = ax.get_yticks().tolist()
    ytlabs = [format(y,'.0f') for y in ytlabs]
    ax.yaxis.set_ticklabels(ytlabs, fontsize=8)

    # Print x and y labels
    ax.set_ylabel(iparams['xlab'], fontsize=10)
    ax.set_xlabel(iparams['ylab'], fontsize=10)
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    return fig,ax
################################################################################################
def gini(array,constant=0):
    '''
    Calculate the Gini coefficient of a numpy array
    '''
    from scipy.stats import rankdata
    # All values are treated equally, arrays must be 1d and float:
    array = np.array(array)
    array = (array.flatten()).astype('float')
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += constant
    # Values must be sorted:
    array = np.sort(array)
    #print(array)
    # Index per array element:
    #index = np.arange(1,array.shape[0]+1)
    index = rankdata(array)
    #print(index)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:

    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))
##############################################################################################

######################################################################################
def norm(x):
    """Dot product-based Euclidean norm implementation

    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    """
    return sqrt(squared_norm(x))
######################################################################################
def hoyer(x):
    """Hoyer's measure of sparsity for a vector"""
    sqrt_n = np.sqrt(len(x))
    return (sqrt_n - np.linalg.norm(x, 1) / norm(x)) / (sqrt_n - 1)
######################################################################################
def matrix_sparsity(A):
    '''
    
    '''
    A = np.array(A)
    sparsity = 1.0 - ( np.count_nonzero(A) / float(A.size) )
    
    return sparsity
##############################################################################################
def calc_sparsity(pulse_times,binspikes,spr_method=gini,max_width=250):
    '''
    
    '''
    odata = []
    for t in pulse_times.astype(int):
        if t[1]-t[0] < max_width:
            temp_spk = np.sum(binspikes[t[0]:t[1],:],0)
            if np.sum(temp_spk) > 0:
                odata.append(spr_method(temp_spk))
            else:
                odata.append(np.nan)

    return odata
######################################################################################
def generate_gini_over_time(idata,ntrials=100):
    '''
    
    '''
    odata = []
    for trial in np.arange(ntrials):
        odata.append(gini(idata[:,trial]))
    
    return np.array(odata)
#######################################################################################
def generate_hoyer_over_time(idata,ntrials=100):
    '''
    
    '''
    odata = []
    for trial in np.arange(ntrials):
        odata.append(hoyer(idata[:,trial]))
    
    return np.array(odata)
#######################################################################################

