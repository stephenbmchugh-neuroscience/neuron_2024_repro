from peakdetect import detect_peaks
import numpy as np
import os as os
import time as time 
import sys as sys 
import pandas as pd
from importlib import reload as reload
import scipy.signal as sig
from scipy import io
import scipy.ndimage as ndimage

# any bugs? please talk to vitor

sr = 1250.
int16 = np.int16
intan2uV = 0.195 

def LoadUnits(b, par=None):
        '''Load "units" information (mostly from des-file).

        INPUT:
        - [b]:       <str> containing "block base"

        OUTPUT:
        - [trodes]:  <DataFrame>'''

        ## If not provided, load the par-file information
        if par is None:
                par = LoadPar(b)

        ## For each tetrode, read in its "per tetode" des-file
        trode_index = range(1, len(par['trode_ch'])+1)
        units = [pd.read_csv(b+'.des.'+str(t), header=None, names=['des']) for t in trode_index]
        units = pd.concat(units, keys=trode_index, names=['trode','trode_unit']).reset_index()
        # -as a check, also read in the "overall" des-file
        all_trodes = pd.read_csv(b+'.des', header=None, names=['des'])
        if ~np.all(all_trodes.des == units.des):
                #print('tetrode des files do not match combined des for '+b)
                units.des = all_trodes.des

        ## Llet the "index per tetrode" and the index of this <DataFrame> start from 2(!) instead of 0
        units['trode_unit'] += 2
        units.index += 2

        ## Add "unit" as column, and set name of column-index to "unit" (NOTE: not sure why this is needed!?)
        #units['unit'] = units.index
        #units.index.set_names('unit', inplace=True)

        ## Return the "unit"-information as <DataFrame>
        return units

def LoadDesv(b):
        desv = pd.read_csv(b+'.desv',index_col=0)
        trodes = LoadTrodes(b)
        if 'hm' in trodes.keys():
                desv = addUnitHM(desv,trodes)
                
        if 'position' in trodes.keys():
                desv['pos'] = getupos(desv,trodes)

        return desv
    
def LoadSpikeTimes(b, trode=None, MinCluId=2, res2eeg=(1250./20000)):

        t = '' if trode is None else '.'+str(int(trode))

        #res = pd.read_csv(b+'.res'+t, header=None, squeeze=True).values
        #clu = pd.read_csv(b+'.clu'+t, squeeze=True).values
        res = pd.read_csv(b+'.res'+t, header=None).squeeze('columns').values
        clu = pd.read_csv(b+'.clu'+t).squeeze('columns').values # this was changed as squeeze=True will be removed from this function in future updates

        if MinCluId is not None:
                mask = clu >= MinCluId
                clu = clu[mask]
                res = res[mask]
        res = np.round(res*res2eeg).astype(int)

        return res,clu

def LoadStages(b, par=None):
    '''Load "stages" information (mostly from desen- and resofs-file).

    INPUT:
    - [b]:       <str> containing "block base"

    OUTPUT:
    - [stages]:  <DataFrame>'''

    ## If not provided, load the par-file information
    if par is None:
        par = LoadPar(b)
    
    ## Read desen- and resofs-file and store as <DataFrame> using "pandas"-package
    stages = pd.read_csv(b+'.desen', header=None, names=['desen'])
    
    try:
	    resofs_ = pd.read_csv(b+'.resofs', header=None)
	    
	    ## Add start- and end-time and filebase of each session to the "stages"-<DataFrame>
	    stages['start_t'] = [0] + list(resofs_.squeeze().values)[:-1]
	    stages['end_t'] = resofs_
    except:
	    1
    stages['filebase'] = par['sessions']

    ## Let the index of this <DataFrame> start from 1 instead of 0
    stages.index += 1

    ## Add "stage" as column, and set name of column-index to "stage" (NOTE: not sure why this is needed!?)
    #stages['stage'] = stages.index
    #stages.index.set_names('stage', inplace=True)

    ## Return the "stages"-information as <DataFrame>
    return stages

def LoadPar(b):

        '''LoadPar
        Parses block or stage level par file into <dict>-object.

        INPUT:
        - [b]:   <str> containing "block base" (= path + file-base; e.g., '/mnfs/swrd3/data/ddLab_merged/mdm96-2006-0124/mdm96-2006-0124')

        OUTPUT:
        - [par]: <dict> containing important info from par-file
                   - 'nch' = <int> total number of recorded channels
                   - 'ref_ch' = <int> channelID of 'reference'
                   - 'ref_trode' = <int> tetrode-number of 'reference'
                   - 'trode_ch' = <list> with for each tetrode a <list> with its channelIDs
                   - 'sessions' = <list> with all session-names'''

        ## Read par-file, returns a <list> with each row converted into a <str>
        lines = open(b+'.par').readlines()

        ## Create an "anonymous" function to split a <str> into its constituent integers
        #to_ints = lambda x:map(int, x.split())

        ## Extract total number of channels, number of tetrodes and channelID of "reference"
        nch, bits = np.array(lines[0].split()).astype(int)
        num_trodes, ref_ch = np.array(lines[2].split()) #.astype(int)
        num_trodes = int(num_trodes)

        ## Create <list> with for each tetrode a <list> with its  channelIDs
        trode_ch = []
        # -loop over all tetrodes
        for l in lines[3:3+num_trodes]:
            l = np.array(l.split()).astype(int) #to_ints(l)
            t = l[1:]
            # -check whether for this tetrode correct number of channels is listed in par-file
            assert l[0]==len(t), 'par error: n ch in trode'
            trode_ch.append(t)

        ## Find tetrode-number of "reference"
        for tetrodeIndex in range(1, num_trodes+1):
            if type(ref_ch) is int:
            	if ref_ch in trode_ch[tetrodeIndex-1]:
                	ref_trode = tetrodeIndex
            	else:
                	ref_trode = 'not specified'

        ## Create <list> with all session-names
        sessions = list(map(str.strip, lines[4+num_trodes:]))
        # -check whehter correct number of sessions is listed in par-file
        assert int(lines[3+num_trodes])==len(sessions), 'par error: n sessions'

        ## Create <dict>-object and return it
        if 'ref_trode' in globals():
            	par = {'nch':nch, 'ref_ch':ref_ch, 'ref_trode':ref_trode, 'trode_ch':trode_ch, 'sessions':sessions}
        else:
            	par = {'nch':nch, 'ref_ch':ref_ch, 'trode_ch':trode_ch, 'sessions':sessions}
        return par

def LoadTrodes(b, par=None):
        '''Load "trodes" information (mostly from desel- and par-file).

        INPUT:
        - [b]:       <str> containing "block base"

        OUTPUT:
        - [trodes]:  <DataFrame>'''

        ## If not provided, load the par-file information
        if par is None:
            par = LoadPar(b)

        ## Read desel-file and store as <DataFrame> using "pandas"-package
        trodes = pd.read_csv(b+'.desel', header=None, names=['desel'])

        ## Add for each tetrode its # of channels, the channelID of its 
        ## "main" channel and a list of all its channelIDs to the "trodes"-<DataFrame>
        trodes['n_tr_ch'] = list(map(len, par['trode_ch']))

        aux = [pari for (pari,par_) in enumerate(par['trode_ch']) if np.size(par_)==0]
        for aux_ in aux:
                par['trode_ch'][aux_] = np.array([-1])
        trodes['lfp_ch'] = list(map(lambda x: x[0], par['trode_ch']))

        ## Let the index of this <DatLoadStagesaFrame> start from 1 instead of 0
        trodes.index += 1

        ## Add "trode" as column, and set name of column-index to "trode" (NOTE: not sure why this is needed!?)
        #trodes['trode'] = trodes.index
        #trodes.index.set_names('trode', inplace=True)

        trodes['desel'] = trodes['desel'].astype(str)
        
        if os.path.exists(b+'.info.el'):
                el = pd.read_pickle(b+'.info.el')
                
                try:
                        hm = np.zeros(len(trodes),dtype='str')
                        for teti,tet in enumerate(trodes.index):
                                hm[teti] = el['hm'].values[el['tet'].values==tet][0]
                except:
                        print('error: couldnt read data from el pickle')
                        hm = ['?']*len(trodes)
                
                trodes['hm'] = hm
                
        if os.path.exists(b+'.desel.pos'):
                position = pd.read_csv(b+'.desel.pos')['position'].values
                trodes['position'] = position
                
        ## Return the "trodes"-information as <DataFrame>
        return trodes

def LoadIntervals(b,ext,toeeg=True):

        ## Read in the file, store as <DataFrame> and return
        interval = pd.read_csv(b+ext, sep='\s+', header=None,names=['begin','end'])
        if toeeg:
            interval['begin'] = (interval['begin']*(1250./20000)).astype(int)
            interval['end'] = (interval['end']*(1250./20000)).astype(int)
        else:
            interval['begin'] = (interval['begin']).astype(int)
            interval['end'] = (interval['end']).astype(int)
        return interval

def bfrombs(bs):
        return bs[:np.max([i for (i,char_) in enumerate(bs) if char_=='_'])]

def loadlfps(bs,usedesel=True):
    
        aux = np.max([i for (i,val) in enumerate(bs) if val=='_'])
        b = bs[:aux]
        
        par = LoadPar(b)
        trodes = LoadTrodes(b)
        path = bs+'.eeg'
        lfps_ = MapLFPs(path,par['nch'])
        if usedesel:
                trodes = LoadTrodes(bfrombs(bs))
                lfps_ = lfps_[trodes['lfp_ch'],:]
                lfps_[trodes['lfp_ch']<0,:] = 0
        return lfps_

def MapLFPs(path, nch, dtype=int16,order='F'):
        '''Returns a 2D numpy <memmap>-object to a binary file, which is indexable as [channel, sample].

        INPUT:
        - [path]:              <str> containing full path to binary-file
        - [nch]:               <int> number of channels in binary file
        - [dtype]=np.int16:    <numpy-type> of binary data points'''

        ## Calculate the total number of data points in the provided binary-file
        size = os.path.getsize(path)
        size = int(size/np.dtype(dtype).itemsize)

        ## Create and return the 2D memory-map object
        memMap = np.memmap(path, mode='r', dtype=dtype, order=order, shape=(nch, int(size/nch)))

        return memMap

def getsessiondur(stages):
    
        endt = stages['end_t']
        startt = stages['start_t']
        sdur = ((endt-startt).values/16).astype(int)
        
        return sdur

def readfoldernm(folder):
    
        if folder[-1]=='/':
                folder = folder[:-1]
        aux = np.max([i for (i,char_) in enumerate(folder) if char_ == '/'])
        recday = folder[1+aux:]         
        b = folder+'/'+recday
    
        return b,recday

def resofs(b,dtype=np.int16,order='F'):
        
        ss = LoadStages(b).index
        par = LoadPar(b)
        nch = int(par['nch'])
        
        resofs_ = np.zeros(len(ss))
        for (si,s) in enumerate(ss):
                bs = b+'_'+str(s)
                path = bs+'.eeg'

                size = os.path.getsize(path)
                size = int(size/np.dtype(dtype).itemsize)
                resofs_[si] = size/(nch*1250)
                
        return resofs_

def loadColors(n,cmap,shuffle=False):

        import matplotlib.pyplot as plt
        colors = plt.get_cmap(cmap)(np.linspace(1,0,n))
        if shuffle:
                colors = colors[np.argsort(np.random.rand(n))]

        return colors
    
def getREMcy(cyl,acc_,accthrs,ratio1=.8,ratio2=.8,prewin_s=120,sr=sr):
    
        # ratio1: samples below acc thrs during theta chain
        # ratio2: samples below acc thrs prewin_s before theta chain onset

        remcy = np.array([],dtype=int).reshape(0,6)
        for cygroupi,cy_ in enumerate(cyl):

                accratio = np.mean(acc_[cy_[:,1]]<accthrs)
                preaccratio = np.mean(acc_[cy_[0,1]+np.arange(-prewin_s*sr,0,dtype=int)]<accthrs)
                if (accratio>ratio1)&(preaccratio>ratio2):
                        remcy = np.concatenate((remcy,cy_))
                        
        return remcy

def save(filename,data,pickle=False):
    
        if type(data) is dict:
                pickle = True
        
        if pickle:
                import pickle
                with open(filename, "wb") as fp:   #Pickling
                        pickle.dump(data, fp)
        else:
                output = open(filename, 'wb')
                np.save(output,data)
                output.close()
                
def load(filename,dopickle=False):
        data = np.load(filename,allow_pickle=True)
        return data
    
def fmaxage_run(fname,fmaxage,return_age=False):
    
        fname_age = np.nan
        if os.path.exists(fname):
                fname_age = getfileage(fname)
                cond = fname_age>fmaxage
        else:
                cond = True
                
        if return_age:
                return cond,fname_age
        else:
                return cond

def savefig(filename,format_='svg'):
        import matplotlib.pyplot as plt
        from vPlotFunctions import config
        config()
        plt.savefig(filename+'.'+format_, format=format_)

def ThetaFilter(RawLFP,SamplingRate=sr,ThetaLow=4.,ThetaHigh=16.,FilterOrder=4,axis=-1):
    
        ThetaLowNorm = (2./SamplingRate)*ThetaLow # Hz * (2/SamplingRate)
        ThetaHighNorm = (2./SamplingRate)*ThetaHigh # Hz * (2/SamplingRate)
    
        # filtering
        b,a = sig.butter(FilterOrder,[ThetaLowNorm,ThetaHighNorm],'band') # filter design
        Theta = sig.filtfilt(b,a,RawLFP,axis=axis)

        return Theta

def bandpass(signal,lowf,hif,sr=sr,FilterOrder=4,axis=-1):
    
        lowf_ = (2./sr)*lowf # Hz * (2/SamplingRate)
        hif_ = (2./sr)*hif # Hz * (2/SamplingRate)
    
        # filtering
        b,a = sig.butter(FilterOrder,[lowf_,hif_],'band') # filter design
        output = sig.filtfilt(b,a,signal,axis=axis)

        return output

def highpass(signal,fcut,sr=1250.,FilterOrder=4,axis=-1):

        hif_ = (2./sr)*fcut # Hz * (2/SamplingRate)
    
        # filtering
        b,a = sig.butter(FilterOrder,hif_,'high') # filter design
        output = sig.filtfilt(b,a,signal,axis=axis)

        return output

def lowpass(signal,fcut,sr=1250.,FilterOrder=4,axis=-1):

        lof_ = (2./sr)*fcut # Hz * (2/SamplingRate)
    
        # filtering
        b,a = sig.butter(FilterOrder,lof_,'low') # filter design
        output = sig.filtfilt(b,a,signal,axis=axis)

        return output
    
#########################################################################################################
#### PEAK DETECTION #####################################################################################
################ __author__ = "Marcos Duarte, https://github.com/demotu/BMC"        #####################
################     ____version__ = "1.0.4"                                        #####################
################     ____license__ = "MIT"                                          #####################
#########################################################################################################

def detectPeaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

#########################################################################################################
#########################################################################################################

def defineThetaCycles(Theta,lowFreqAmp,SamplingRate=1250.):

	# Theta: theta component of your signal (might be EMD-based)
	# ... for EEMD approach -> 'auxsignal' must be the low-frequency component envelope that will be 
	#			        used as amplitude threshold in STEP 2 below.
	#                          'lowfreqCutOff' must be str 'emd'

	###################################################################################################
	# STEP 1. Define some parameters ################################################################
	# below define what is be the minimum peak to peak (and valley to valley) interval allowed
	MinPeak2PeakDistance = int(round((1./15)*SamplingRate)) # I use a 15-Hz cycle

	# below define what is be the maximum peak to peak (and valley to valley) interval allowed
	MaxPeak2PeakDistance = int(round((1./5)*SamplingRate)) # I use a 5-Hz cycle

	# below define what is the maximum and minimum peak to valley internval
	MinPeak2ValleyDistance = int(round((1./16)*SamplingRate/2)) # 16-Hz half-cycle for minimum 
	MaxPeak2ValleyDistance = int(round((1./4)*SamplingRate/2)) # 4-Hz half-cycle for maximum

	###################################################################################################
	# STEP 2. Define thresholds for peak detection ####################################################

	# I use two threholds.

	# FIRST THRESHOLD is based on slow oscillations oscillations. 
	# The rationale is that we want real theta oscillations and not 1/f signals. 
	# So theta peaks have to be larger than low-frequency amplitudes.

	lowfreqthrs = np.copy(lowFreqAmp)
	
	# SECOND THRESHOLD is a fixed (arbitrary) threshold
	# I'm looking still for a better definition.
	MinThetaPeakAmplitude = np.median(np.abs(Theta))*.25
	# this is to avoid to get periods where the signal is (nearly) flat

	# line below combines both thresholds
	lowfreqthrs[lowfreqthrs<MinThetaPeakAmplitude] = MinThetaPeakAmplitude	

	###################################################################################################
	# STEP 3. Theta peak and valley detection #########################################################

	# I use this function for peak detection. It is in ddLab
	PeakIs = detect_peaks(Theta,show=False,mph=MinThetaPeakAmplitude,mpd=MinPeak2PeakDistance)
	ValleyIs = detect_peaks(-Theta,show=False,mph=MinThetaPeakAmplitude,mpd=MinPeak2PeakDistance)

	# then, take only peaks and valleys that pass amplitude thresholds
	PeakIs = PeakIs[Theta[PeakIs]>=lowfreqthrs[PeakIs]]
	ValleyIs = np.unique(ValleyIs[Theta[ValleyIs]<=-lowfreqthrs[ValleyIs]])

	###################################################################################################
	# STEP 4. Definitions of Theta cycles #############################################################

	# in order to 'detect' a theta cycle I go valley by valley and check if the preceding AND the subsequent 
	# detected peaks are within a distance compatible with the theta cycle length (following the parameters 
	# defined in the first step), c.f. the loop below.

	# declaring variables as empty int arrays
	ThetaCycleBegin = np.array([],dtype=int)
	ThetaCycleEnd = np.array([],dtype=int)
	CycleRefs = np.array([],dtype=int)

	# loop over valleys
	for Valleyi in ValleyIs:
	    
		# gets first peak BEFORE valley
		aux = PeakIs[(PeakIs>(Valleyi-MaxPeak2ValleyDistance))&(PeakIs<Valleyi)]
		if np.size(aux)>0:
			Peak1 = aux[np.argmax(aux)]
		else:
			Peak1 = -np.inf
	    
		# gets first peak AFTER valley
		aux = PeakIs[(PeakIs<(Valleyi+MaxPeak2ValleyDistance))&(PeakIs>Valleyi)]
		if np.size(aux)>0:
	    		Peak2 = aux[np.argmin(aux)]
		else:
			Peak2 = -np.inf
	    
		# checks if both peak-valley distances are larger than minimum allowed
		PeakValleyCheck1 = min((Valleyi-Peak1),(Peak2-Valleyi))\
		                    >=MinPeak2ValleyDistance
		# checks if both peak-valley distances are smaller than maximum allowed
		PeakValleyCheck2 = max((Valleyi-Peak1),(Peak2-Valleyi))\
		                    <=MaxPeak2ValleyDistance
    	    
		# if both conditions are satisfied, get theta cycle
		if PeakValleyCheck1&PeakValleyCheck2:
			if (Peak2-Peak1)<=MaxPeak2PeakDistance:
				ThetaCycleBegin = np.append(ThetaCycleBegin,Peak1)
				ThetaCycleEnd = np.append(ThetaCycleEnd,Peak2)
				CycleRefs = np.append(CycleRefs,Valleyi)

	CyclePeaks = np.asarray([ThetaCycleBegin,ThetaCycleEnd]).T

	# gets rid of cycles with coincident first edges.
	aux = list(CyclePeaks[:,0])
	cycles2remove = list(set([i for (i,x) in enumerate(aux) if aux.count(x) > 1]))
	del aux

	CyclePeaks = np.delete(CyclePeaks,cycles2remove,0)
	CycleTrough = np.delete(CycleRefs,cycles2remove,0)

	nCycles = len(CycleTrough)
	zeroscross1 = np.zeros(nCycles,dtype=int)
	zeroscross2 = np.zeros(nCycles,dtype=int)
	zeroscross3 = np.zeros(nCycles,dtype=int)
	validCycles = np.zeros(nCycles,dtype=bool)
	for cyclei in range(nCycles):

		aux1 = np.arange(CyclePeaks[cyclei,0]-MaxPeak2ValleyDistance,CyclePeaks[cyclei,0],dtype=int)
		aux2 = np.arange(CyclePeaks[cyclei,0],CycleTrough[cyclei],dtype=int)
		aux3 = np.arange(CycleTrough[cyclei],CyclePeaks[cyclei,1],dtype=int)

		cond0 = (np.sum(Theta[aux1]<0)*np.sum(Theta[aux2]<0)*np.sum(Theta[aux3]>0))>0
		if cond0:
			zeroscross1[cyclei] = np.max(np.where(Theta[aux1]<0))+1+(CyclePeaks[cyclei,0]-MaxPeak2ValleyDistance)
			zeroscross2[cyclei] = np.min(np.where(Theta[aux2]<0))+(CyclePeaks[cyclei,0])-1
			zeroscross3[cyclei] = np.min(np.where(Theta[aux3]>0))+(CycleTrough[cyclei])-1

			cond1 = np.sum(Theta[np.arange(zeroscross3[cyclei]+1,CyclePeaks[cyclei,1])]<0)<1
			cond2 = np.sum(Theta[np.arange(zeroscross1[cyclei]+1,CyclePeaks[cyclei,0])]<0)<1
			cond3 = np.sum(Theta[np.arange(zeroscross2[cyclei]+1,CycleTrough[cyclei])]>0)<1
			validCycles[cyclei] = cond1&cond2&cond3

	cycleRefs = np.array([zeroscross1,CyclePeaks[:,0],zeroscross2,CycleTrough,zeroscross3,CyclePeaks[:,1]]).T
	cycleRefs = cycleRefs[validCycles,:]

	return cycleRefs

def DeltaFilter(RawLFP,SamplingRate=1250.,cutoff=5,FilterOrder=4):
    
	cutoff = (2./SamplingRate)*cutoff # Hz * (2/SamplingRate)
    
	# filtering
	b,a = sig.butter(FilterOrder,cutoff,'lowpass') # filter design
	Theta = sig.filtfilt(b,a,RawLFP)
	
	return Theta

def TortModIndex(meanAmp):
        nph = np.size(meanAmp,0)
        normMeanAmp = meanAmp/np.kron(np.ones((nph,1)), np.sum(meanAmp,axis=0))
        logNBins = np.log2(nph)
        
        HP = -sum(normMeanAmp*np.log2(normMeanAmp))
        Dkl = logNBins-HP
        
        return Dkl/logNBins

def runAmp2Ph(spect,phase,phaseedges):

        reclen = len(phase)
        spect = spect.reshape(-1,reclen)
        nPh = len(phaseedges)-1
        ampphase = np.zeros((len(spect),nPh)) + np.nan
        for phbi in range(nPh):
                phsmps = (phase>=phaseedges[phbi])&(phase<=phaseedges[phbi+1])
                ampphase[:,phbi] = np.mean(spect[:,phsmps],axis=1)
                
        return ampphase

def getThetaPhases(cycleRefs,signalLen):

        phase = np.zeros(signalLen)+np.nan
        nCycles = np.size(cycleRefs,0)
        for cyclei in range(nCycles):
                phaserefs = cycleRefs[cyclei,:]
                for phaserefi in range(len(phaserefs)-2):
                        initialphase = (-np.pi/2)+(np.pi/2)*phaserefi
                        endphase = (-np.pi/2)+(np.pi/2)*(phaserefi+1)
                        quadrantSamples = range(phaserefs[phaserefi],phaserefs[phaserefi+1])
                        quadrantTimeLength =  len(quadrantSamples)
                        phase[quadrantSamples] = np.linspace(initialphase,endphase,quadrantTimeLength+1)[0:-1]

        validPhases = np.where(~np.isnan(phase))[0]
        phase[validPhases[phase[validPhases]<0]] += 2*np.pi

        return phase

def binEdges2Centres(BinEdges):

	BinCenters = np.convolve(BinEdges,[.5,.5],'same')
	BinCenters = BinCenters[1::]

	return BinCenters

def hist(data,bins):
    
        data = data[~np.isnan(data)]

        Counts,BinEdges = np.histogram(data,bins)
        BinCenters = binEdges2Centres(BinEdges)
        
        return Counts,BinCenters,BinEdges

def getIMFmainfreq(IA,IF):

	nimfs = np.size(IA,0)

	mainfreqs = np.zeros(nimfs)
	for imfi in range(nimfs):

                if0 = np.copy(IF[imfi,1:-1]) #np.copy(IF[1:,imfi])
                ia0 = np.copy(IA[imfi,1:-1]) #np.copy(IA[1:,imfi])

                mainfreqs[imfi] = np.sum(if0*pow(ia0,2))/np.sum(pow(ia0,2))

	return mainfreqs

def normrows(matrix,op=np.min,opsub=None,refmat=None):
    
        if refmat is None:
                refmat = matrix
    
        divmat = op(refmat,axis=1)
        divmat = np.tile(divmat,(np.size(matrix,1),1)).T
        
        if opsub is None:
                submat = np.zeros_like(matrix)
        else:
                submat = opsub(refmat,axis=1)
                submat = np.tile(submat,(np.size(matrix,1),1)).T

        nmatrix = np.zeros_like(matrix)+np.nan
        nmatrix[divmat[:,0]>0] = (matrix[divmat[:,0]>0]-submat[divmat[:,0]>0])/divmat[divmat[:,0]>0]
        
        return nmatrix
    
def bootstrap_ci(data,nsamps=10,nboots=100,alpha=[0.01],replace=False):
    
        np.random.seed()
    
        n = len(data)
    
        mdata = np.zeros((nboots,np.size(data,1)))
        for booti in range(nboots):
            
                draw = np.random.choice(n,nsamps,replace=replace)
                mdata[booti] = np.mean(data[draw],axis=0)
                
        ci = np.zeros((len(alpha),2,np.size(mdata,1)))
        for alphai,alpha_ in enumerate(alpha):
            
                alpha_ = 100*alpha_/2
                ci[alphai] = np.percentile(mdata,[alpha_,100-alpha_],axis=0)
                
        ci = ci.squeeze()
                
        return ci

def computeActMat(res,clu,cluIds='all',units=None,ncols=None):

        if type(cluIds) is str:
                if cluIds == 'all':
                        if units is None:
                                cluIds = np.unique(clu)
                        else:
                                cluIds = units.index
                elif not(units is None):
                        cluIds = np.array(units['desv'].index[units['desv']==cluIds])
        indexes = np.in1d(clu,cluIds)
        clu = clu[indexes]
        res = res[indexes]
        
        nclu = len(cluIds)
        
        if ncols is None:
                ncols = np.max(res)+1
        
        bins = np.arange(0,ncols+1) # this will be changed for different binsizes
        actMat = np.zeros((nclu,ncols),dtype=int)
        for (clui,cluid) in enumerate(cluIds):
                spks = res[clu==cluid]
                actMat0,_,_ = hist(spks,bins)
                actMat[clui,:] = actMat0
                
        return actMat,cluIds

def triggeredAverage(sig2trig,trigger,taLen=500,sr=1250.,average=True):
    
        taLen = int(taLen)
        prepost = np.arange(taLen,dtype=int)-int(taLen/2)

        if len(np.shape(sig2trig))==1:
                sig2trig = sig2trig[None,:]
        
        mask_trig = trigger<(np.size(sig2trig,1)+np.min(prepost))
        trigger = trigger[mask_trig]
              
        if average:
                ta = np.zeros((np.size(sig2trig,0),taLen))
                for t in trigger:
                        ta += sig2trig[:,t+prepost]
                ta /= len(trigger)
                ta = ta.squeeze()
        else:
                ta = np.zeros((len(mask_trig),np.size(sig2trig,0),taLen)) + np.nan
                tis = np.where(mask_trig)[0]
                for (tii,t) in enumerate(trigger):
                        ti = tis[tii]
                        ta[ti] = sig2trig[:,t+prepost]

        taxis = 1000.*prepost/sr
        
        return ta,taxis

def getActMat(res,clu,binage=125,units=None):

        if type(binage) is int:
                binedges = np.arange(0,np.max(res)+binage,binage)
        elif type(binage) is np.ndarray:
                binedges = binage
        else:
                print('ERROR!!! binage type not understood')
                return None,None,None

        nbins = len(binedges)-1

        if units is None:
                cluids = np.unique(clu)
        else:
                cluids = units.index
                
        nc = len(cluids)
        actmat = np.zeros((nc,nbins))
        for (ci,c) in enumerate(cluids):
                spks = res[clu==c]
                actmat[ci,:],bincentres,_ = hist(spks,binedges)
        
        bincentres = bincentres
        
        return actmat,bincentres,cluids

def getCycleActMat(res,clu,cycles,units,cols=[0,4]):

        nunits = len(units)
        ncycles = len(cycles)
    
        actmat = np.zeros((nunits,ncycles))
        for uniti,cluid in enumerate(units.index):

                spktimes = res[clu==cluid]

                cycleedges = cycles[:,np.array(cols)]
                cycleedges[:,1] -= 1
                cycleedges = cycleedges.reshape(-1)

                actmat_,_, = np.histogram(spktimes,cycleedges)

                actmat[uniti,:] = actmat_[np.arange(0,len(cycleedges),2)] 
                
        return actmat
    
def addUnitHM(units,trodes):

        units['hm'] = np.zeros(len(units),dtype='str')
        for ti,te in enumerate(trodes.index):
                hm = trodes.hm[te]
                umask = units.trode.values==te
                units['hm'].values[umask] = hm
                
        return units

def xcorr(a,b,maxlag,step=1):
    
        import pandas as pd
        
        datax = pd.Series(b)
        datay = pd.Series(a)
    
        lags = np.arange(-maxlag,maxlag,step)
        output = np.zeros(len(lags))
        lagi = -1
        for lag in lags:
                lagi += 1
                output[lagi] = datax.corr(datay.shift(lag))
                
        return output,lags
    
def xcorr(a, b, max_lags):
        """
        Compute the Pearson correlation coefficient (r) between two vectors a and b using FFT with a given max lags.

        Parameters:
        a (numpy array): First input vector.
        b (numpy array): Second input vector.
        max_lags (int): Maximum lags for the cross-correlation.

        Returns:
        numpy array: Pearson correlation coefficient values for the given max lags.
        """

        a = np.asarray(a)
        b = np.asarray(b)
        
        # Compute standard deviations
        std_a = np.std(a)
        std_b = np.std(b)
        
        # Generate the lag values
        lags = np.arange(-max_lags, max_lags + 1)
        
        if (std_a>0)&(std_b>0):

                # Calculate the length for zero-padding (next power of 2 for efficiency)
                n = 2 ** int(np.ceil(np.log2(len(a) + len(b) - 1)))

                # Compute the FFT of the zero-padded vectors
                A = np.fft.fft(a - np.mean(a), n=n)
                B = np.fft.fft(b - np.mean(b), n=n)

                # Compute the cross-correlation using the inverse FFT of the product
                corr_full = np.fft.ifft(A * np.conj(B)).real

                # Roll the result to align the zero lag in the center
                corr_full = np.fft.fftshift(corr_full)

                # Calculate the middle index
                mid = len(corr_full) // 2

                # Slice the result to get the cross-correlation for the given max lags
                corr = corr_full[mid - max_lags: mid + max_lags + 1]

                # Normalize the cross-correlation to convert to Pearson's r
                normalization_factor = len(a) * std_a * std_b
                xcor = corr / normalization_factor
                
        else:
                xcor = np.zeros(len(lags))+np.nan

        return xcor, lags
    
def detect_OscBursts(lfp,minncy = 8, band=[20,30], surroundBand = [15,60], thrs_ = 1.5, sr=sr,thrs=None):

        lfpf = bandpass(lfp,band[0],band[1])

        lfpf_hil = runHilbert(lfpf)
        lfpf_env = np.abs(lfpf_hil)
        lfpf_ph = np.angle(lfpf_hil)
        
        lfpf_surround = bandpass(lfp,surroundBand[0],surroundBand[1])
        lfpf_surround_env = np.abs(runHilbert(lfpf_surround))

        candidates = detect_peaks(lfpf_env,show=False,mph=0,mpd=int(minncy*50*sr/1000)) 

        if thrs is None:
                thrs = thrs_*np.diff(np.percentile(lfpf_env[candidates],[25,75]))\
                                                    +np.percentile(lfpf_env[candidates],75)
        thrsWin = thrs/2

        candidates = candidates[lfpf_env[candidates]>thrs]

        timestamps = np.array([],dtype=int).reshape(0,3)
        features = {'peakamp': np.array([]),\
                    'ncycles': np.array([],dtype=int),\
                    'meanfreq': np.array([]),\
                    'dur': np.array([]),\
                    'power': np.array([]),\
                    'surpower': np.array([]),\
                    'powerratio': np.array([])}   
        
        par = {'thrs': thrs,\
                    'thrsWin': thrsWin,\
                    'lfpf': lfpf}   
        
        for (ci,cx) in enumerate(candidates):

                if np.sum(lfpf_env[:cx]<thrsWin)>0:
                        onset = np.max(np.where(lfpf_env[:cx]<thrsWin)[0])
                else:
                        onset = 0

                if np.sum(lfpf_env[cx:]<thrsWin)>0:
                        offset = cx + np.min(np.where(lfpf_env[cx:]<thrsWin)[0])
                else:
                        offset = len(lfpf_env)-1

                eventwin = np.array([onset,offset])
                eventsamples = np.arange(eventwin.min(),eventwin.max()+1)

                phase = np.copy(np.unwrap(lfpf_ph[eventsamples]))
                phase -= phase[0]
                ncycles = phase[-1]/(2*np.pi)

                power = pow(np.sum(lfpf_env[eventsamples]),2)/1250.
                surpower = pow(np.sum(lfpf_surround_env[eventsamples]),2)/1250.
                
                powerratio = power/surpower
                    
                if (ncycles>=minncy)&(powerratio>0):

                        meanfreq = ncycles/(len(eventsamples)/1250)

                        aux = np.array([onset,cx,offset]).reshape(1,3)
                        timestamps = np.concatenate((timestamps,aux))
                        
                        features['peakamp'] = np.concatenate((features['peakamp'],[lfpf_env[cx]]))
                        features['ncycles'] = np.concatenate((features['ncycles'],[ncycles]))
                        features['meanfreq'] = np.concatenate((features['meanfreq'],[meanfreq]))
                        features['dur'] = np.concatenate((features['dur'],[len(eventsamples)/1250.]))
                        features['power'] = np.concatenate((features['power'],[power]))
                        features['surpower'] = np.concatenate((features['surpower'],[surpower]))
                        features['powerratio'] = np.concatenate((features['powerratio'],[powerratio]))

                ##########

        if len(timestamps)>1:
            
                _,uniqueis = np.unique(timestamps[:,np.array([0,2])],axis=0,return_index=True)

                timestamps = timestamps[uniqueis,:]

                for key in features.keys():
                        features[key] = features[key][uniqueis]
                        
        if len(timestamps)>0:

                peaks = getPeaksWithinWindows(lfpf_env,timestamps[:,np.array([0,2])])
                timestamps[:,1] = peaks
                
        return timestamps,features,par

def wvfilt(signal,mfreq,sr=1250., w=5, s=1, normwv=True):

        # computing the length of the wavelet for the desired frequency
        wavelen = int(np.round(2*w*s*sr/mfreq))

        # constructs morlet wavelet with given parameters
        wave = sig.morlet(wavelen,w=w,s=s,complete=True)

        # cutting borders
        cumulativeEnvelope = np.cumsum(np.abs(wave))/np.sum(np.abs(wave))
        Cut1 = next(i for (i,val) in enumerate(cumulativeEnvelope[::-1]) if val<=(1./2000)) 
        Cut2 = Cut1
        Cut1 = len(cumulativeEnvelope)-Cut1-1
        wave = wave[range(Cut1,Cut2)]

        # normalizes wavelet energy
        if normwv:
                wave = wave/(.5*sum(abs(wave)))
        
        if (len(wave))>len(signal):
                print('ERROR: input signal needs at least '+str(len(wave))+\
                                                            ' time points for '+str(mfreq)+\
                                                                      'Hz-wavelet convolution')
                return None
                
        # convolving signal with wavelet
        fsignal = np.convolve(signal,wave,'same')
        return fsignal


def wvSpect(signal,freqs,tfrout=False,runSpectrogram=True,runPhases=False,s=1,w=5,sr=1250.,normwv=True):

        freqs = np.array(freqs).reshape(-1,1)
        tfr = np.zeros((np.size(freqs),len(signal)),dtype=complex) 
        for (fi,f) in enumerate(freqs):
                f = float(f)
                tfr[fi,:] = wvfilt(signal,f,sr,normwv=normwv) 

        if runSpectrogram:
                output = (np.abs(tfr).squeeze(),)
        if tfrout:
                output += (tfr.squeeze(),)
        if runPhases:
                output += (np.angle(tfr).squeeze(),)

        return output
    
def mvl(phases):
    
        mv = np.mean(np.exp(1j*phases))
        
        mph = np.angle(mv)
        mvl = np.abs(mv)
        
        return mvl,mph
    
def wmvl(phbins,phw):
    
        mv = np.average(np.exp(1j*phbins),weights=phw)
        
        mph = np.angle(mv)
        mvl = np.abs(mv)

        return mvl,mph
    
def get_spkph_hist(spkphs,allphs,nphbins = 32, smoo = 2):
        
        phedges = np.linspace(0,2*np.pi,1+nphbins)

        allph_hist = hist(allphs,phedges)[0]
        count,phbins,_ = hist(spkphs,phedges)

        phprob = count/allph_hist
        
        if smoo>0:
                phprob = circularSmooth(phprob,smoo,int(smoo*2))
        
        return phprob,phbins
    
def get_phpaired_smps(smps, phase, nphbins = 25):
    
        np.random.seed()

        smp_theph = phase[smps]

        counts,phbinc,phbins = hist(smp_theph,np.linspace(0,2*np.pi,1+nphbins))

        smps_control = np.array([],dtype=int)
        for bini in range(nphbins):
                count = counts[bini]
                smp_candidates = np.where((phase>=phbins[bini]) & (phase<phbins[1+bini]))[0]
                paired_smps_ = np.random.choice(smp_candidates, count,replace=False)

                smps_control = np.concatenate((smps_control, paired_smps_))
                
        return smps_control

def runHilbert(signal,axis=-1):

	ndim = len(np.shape(signal))
	if ndim == 1:
		signal = signal[None,:]
	signallen = np.size(signal,1)

	hilbert = sig.hilbert(signal,next_power_of_2(signallen),axis=axis)[:,range(signallen)]
	hilbert = hilbert.squeeze()

	return hilbert

def getPeaksWithinWindows(signal,Edges):

	Edges = Edges[Edges[:,-1]<len(signal),:]
	nWindows = np.size(Edges,0)
	Peaks = np.zeros(nWindows,dtype=int)
	for wi in range(nWindows):
		samples = range(Edges[wi,0],Edges[wi,-1]) 
		Peaks[wi] = samples[np.argmax(signal[samples])]
        
	return Peaks

def getSamplesWithinEdges(Edges):

	if len(np.shape(Edges))<2:
       		Edges = Edges.reshape(1,2)
	samples = np.array([],dtype=int)
	NofCycles = len(Edges)    
	for cyclei in range(NofCycles):
		cyclesamplesAux = np.arange(Edges[cyclei,0],Edges[cyclei,-1],dtype=int)
		samples = np.hstack((samples,cyclesamplesAux))
	samples = np.unique(samples)

	return samples

###############################################################
## TRACKING AND PLACE CELL ANALYSES ###########################

def linearInterpolate(x,kind=None):
    
        x = np.array(x)

        validnumIdxs = np.where(~np.isnan(x))[0]
        validExtrema = [np.min(validnumIdxs),np.max(validnumIdxs)]

        nNANs = np.sum(np.isnan(x[validExtrema[0]:(validExtrema[1]+1)]))

        while nNANs>0:

                nanIdx = np.where(np.isnan(x[validExtrema[0]:(validExtrema[1]+1)]))[0][0]
                nanIdx += validExtrema[0]

                idx1 = np.max(np.where((range(len(x))<nanIdx)\
                                       &(~(np.isnan(x))))[0])
                idx2 = np.min(np.where((range(len(x))>nanIdx)\
                                       &(~(np.isnan(x))))[0])
                
                if kind is not None:
                        from scipy.interpolate import interp1d
                        validx = np.where(~np.isnan(x))[0]
                        interpolator = interp1d(validx,x[validx],kind=kind)
                        x[(idx1+1):idx2] = interpolator(np.arange(idx1+1,idx2))
                else:
                        slope = np.diff(x[[idx1,idx2]])/(idx1-idx2)
                        x[(idx1+1):idx2] = x[idx1]-([1+np.arange(len(x[(idx1+1):idx2]))]*slope)

                nNANs = np.sum(np.isnan(x[validExtrema[0]:(validExtrema[1]+1)]))

        return x
    
def groupConsecutiveVals(x,minN = 1):
    
        edges = np.concatenate(([0],np.where(np.diff(x)>1)[0]+1,[len(x)]))

        groupranges = []
        for groupi in range(len(edges)-1):
            if (edges[1+groupi]-edges[groupi])>minN:
                    aux = np.array([edges[groupi],edges[1+groupi]-1])
                    groupranges.append(x[aux])
                    
        return np.array(groupranges)

trackingSR_Hz = 20000./512

def load_tracking(b, smoothing=1, ext = 'whl'):
        '''Load position data (whl)'''
        trk = pd.read_csv(b+'.' +ext, sep='\s+', header=None).values
        trk[trk<=0] = np.nan
        if (smoothing is not None)&(smoothing>0):
                trk = ndimage.filters.gaussian_filter1d(trk, smoothing, axis=0)
        return pd.DataFrame(trk, columns=['x','y'])
    
def loadTracking(bsnm,smoo=2,mazelen=-1.,maxspeed=-1, ext = 'whl',pixelsPerCm=-1,return_ppc=False):
    
        tracking = load_tracking(bsnm,0,ext)
        #jumpypoints = detectJumpyPoints(bsnm,trackingSR_Hz,mazelen,maxspeed,ext)
        jumpypoints = detectJumpyPoints2(tracking)
        
        tracking['x'][(tracking['x']<=0)|jumpypoints] = np.nan
        tracking['y'][(tracking['y']<=0)|jumpypoints] = np.nan

        if pixelsPerCm>0:
                1
        elif mazelen>0:
                nPixels = np.mean([np.max(tracking['x'])-np.min(tracking['x']),\
                                         np.max(tracking['y'])-np.min(tracking['y'])])
                pixelsPerCm = nPixels/mazelen
        else:
                pixelsPerCm = 1

        if np.mean(np.isnan(tracking['x']))<1:
                tracking['X'] = linearInterpolate(tracking['x'])/pixelsPerCm
                tracking['Y'] = linearInterpolate(tracking['y'])/pixelsPerCm
        else:
                tracking['X'] = tracking['x']/pixelsPerCm
                tracking['Y'] = tracking['y']/pixelsPerCm
                
        jumpypoints = detectJumpyPoints2(tracking,False)
        tracking['X'][(tracking['X']<=0)|jumpypoints] = np.nan
        tracking['Y'][(tracking['Y']<=0)|jumpypoints] = np.nan
        
        if np.mean(np.isnan(tracking['X']))<1:
                tracking['X'] = linearInterpolate(tracking['X'])
                tracking['Y'] = linearInterpolate(tracking['Y'])

        velx = np.diff(tracking['X'])
        vely = np.diff(tracking['Y'])
        speed = np.concatenate((np.array([0]),np.sqrt(pow(velx,2)+pow(vely,2))))
        #speed /= pixelsPerCm
        speed *= trackingSR_Hz
        
        trackingTime = np.arange(0,len(tracking['x']))/trackingSR_Hz
        tracking['speed'] = speed
        tracking['time'] = trackingTime

        if return_ppc:
                return tracking,pixelsPerCm
        else:
                return tracking
    
def detectJumpyPoints2(tracking,rawinput=True,thrs_=4):
    
        if rawinput:
                x = tracking['x'].values.copy()
                y = tracking['y'].values.copy()
        else:
                x = tracking['X'].values.copy()
                y = tracking['Y'].values.copy()
                
        flag = True
        cnt = 0
        jumpypoints = np.array([],dtype=int)
        while flag:
                cnt += 1

                speedx = np.diff(x)
                speedy = np.diff(y)
                speed = np.concatenate((np.array([np.nan]),np.sqrt(pow(speedx,2)+pow(speedy,2))))

                validis = np.where(~np.isnan(speed))[0]

                thrs = np.nanpercentile(speed,75)+np.diff(np.nanpercentile(speed,[25,75]))*thrs_
                outis_ = np.where(speed[validis]>thrs)[0]
                if np.size(outis_)==0:
                        flag = False
                        break

                is_ = np.array([],dtype=int)
                for shift in np.arange(-3,4):
                        aux = outis_+shift
                        aux = aux[aux<len(validis)]
                        is_ = np.concatenate((is_,validis[aux]))
                is_ = np.sort(np.unique(is_))

                jumpypoints = np.concatenate((jumpypoints,is_))

                x[jumpypoints] = np.nan
                y[jumpypoints] = np.nan

                if cnt>5:
                        flag = False
                        break

        jumpypoints_ = np.zeros(len(x),dtype=bool)
        jumpypoints_[jumpypoints] = True
        return jumpypoints_
    
def detectJumpyPoints(bsnm,trackingSR_Hz,mazelen=-1,maxspeed=-1, ext='whl'):

        tracking = load_tracking(bsnm,.1,ext)
        
        if mazelen>0:
                nPixels = np.mean([np.max(tracking['x'])-np.min(tracking['x']),\
                                         np.max(tracking['y'])-np.min(tracking['y'])])
                pixelsPerCm = nPixels/mazelen
        else:
                pixelsPerCm = 1
        
        velx = np.diff(tracking['x'])
        vely = np.diff(tracking['y'])
        speedRaw = np.concatenate((np.array([0]),np.sqrt(pow(velx,2)+pow(vely,2))))
        speedRaw /= pixelsPerCm
        speedRaw *= trackingSR_Hz
        speedRaw[np.isnan(speedRaw)] = -1

        speedRaw_ = speedRaw[speedRaw>0]
        if (np.size(speedRaw_)>10)&(maxspeed <= 0):
                maxspeed = float(np.diff(np.percentile(speedRaw_,[25,75]))*10\
                                                     +np.percentile(speedRaw_,75))
        elif (np.size(speedRaw_)<10)&(maxspeed <= 0):
                maxspeed = np.inf
        
        ids = np.where(speedRaw>maxspeed)[0]
        
        jumpypoints = np.zeros(len(speedRaw),dtype=bool)
        for i in [-1,0,1]:
                jumpypoints[ids+i] = True
                
        return jumpypoints

def MatrixGaussianSmooth(Matrix,GaussianStd,GaussianNPoints=0,NormOperator=np.sum):

	# Matrix: matrix to smooth (rows will be smoothed)
	# GaussiaStd: standard deviation of Gaussian kernell (unit has to be number of samples)
	# GaussianNPoints: number of points of kernell
	# NormOperator: # defines how to normalise kernell

	if GaussianNPoints<GaussianStd:
		GaussianNPoints = int(4*GaussianStd)

	GaussianKernel = sig.get_window(('gaussian',GaussianStd),GaussianNPoints)
	GaussianKernel = GaussianKernel/NormOperator(GaussianKernel)

	if len(np.shape(Matrix))<2:
		SmoothedMatrix = np.convolve(Matrix,GaussianKernel,'same')
	else:
	    	SmoothedMatrix = np.ones(np.shape(Matrix))*np.nan
	    	for row_i in range(len(Matrix)):
	    		SmoothedMatrix[row_i,:] = \
		    		np.convolve(Matrix[row_i,:],GaussianKernel,'same')

	return SmoothedMatrix,GaussianKernel
    
def circularSmooth(matrix,gaussianStd,nPoints,NormOperator=np.sum):
    
    originalLen = np.shape(matrix)[-1]
    matrix = np.hstack((matrix,matrix,matrix,matrix,matrix))
    matrix,_ = MatrixGaussianSmooth(matrix,gaussianStd,nPoints,NormOperator=NormOperator)
    if len(np.shape(matrix))>1:
        matrix = matrix[:,1+originalLen:1+2*originalLen]
    else:
        matrix = matrix[1+originalLen:1+2*originalLen]
    
    return matrix

def runthetaz(array,cycles,sm=0):
    
        'z-scores spike trains (1/1.25 bins) for theta samples only'
    
        thetasamples = getSamplesWithinEdges(cycles[:,np.array([0,4])])

        mean = np.mean(array[:,thetasamples],axis=1)
        std = np.std(array[:,thetasamples],axis=1)
        
        ncols = np.size(array,1)
        zarray = array - np.repeat(mean,ncols).reshape(len(mean),ncols)
        zarray /= np.repeat(std,ncols).reshape(len(mean),ncols)

        if sm>0:
                zarray,_ = MatrixGaussianSmooth(zarray,sm)
                
        return zarray

def printTime(prefix):
	print(prefix+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
	sys.stdout.flush()

def next_power_of_2(n):

	'''Return next power of 2 greater than or equal to n'''
	n -= 1                 # short for "n = n - 1"
	shift = 1
	while (n+1) & n:       # the operator "&" is a bitwise operator: it compares every bit of (n+1) and n, and returns those bits that are present in both
		n |= n >> shift    
		shift <<= 1        # the operator "<<" means "left bitwise shift": this comes down to "shift = shift**2"
	return n + 1

def groupConsec(data,minsize=1):

        from itertools import groupby
        from operator import itemgetter

        groups =[]

        for k,g in groupby(enumerate(data),lambda x:x[0]-x[1]):
                group = (map(itemgetter(1),g))
                group = list(map(int,group))
                
                if np.size(group)>=minsize:
                        groups.append(np.array(group))
                
        return groups 

def runMPs(ps,queue=None):
    
        import multiprocessing as mp

        for p in ps:
                p.start()
        if queue is not None:
                dataout = [queue.get() for p in ps]
        for p in ps:
                p.join()

        if queue is not None:
                return dataout
            
def runMPs_batch(ps,queue=None,maxsim = None):
    
        import multiprocessing as mp
    
        if maxsim is None:
                maxsim = int(mp.cpu_count()*.5)

        nps = len(ps)
        nbatches = int(np.ceil(nps/maxsim))
        batchedges = np.round(np.linspace(0,nps,1+nbatches)).astype(int)

        dataout = []
        for batchi in range(nbatches):

                batchis = np.arange(batchedges[batchi],batchedges[1+batchi])
                ps_batch = [ps[pi] for pi in batchis]

                if queue is None:
                        runMPs(ps_batch,queue)
                else:
                        dataout += runMPs(ps_batch,queue)
                        
        return dataout
    
def loadAcel_(acc,sm=2*1250):
    
        smnpoints = int(sm*4)
        
        accs = np.zeros(np.size(acc,1))+np.nan
        if np.size(acc,1)>(2*smnpoints):
    
                acc = highpass(acc,.1)
                acc = np.sum(pow(acc,2),axis=0)
                acc = np.sqrt(acc)

                accs,skernel = MatrixGaussianSmooth(acc,sm,smnpoints)
                accs[:len(skernel)] = np.nan
                accs[-len(skernel):] = np.nan
        
        return accs
    
def hist2(data1,data2,nbins):

	hist, xedges, yedges = np.histogram2d(data1,data2,nbins)
	xBinCenters = np.convolve(xedges,[.5,.5],'same')
	xBinCenters = xBinCenters[1::]
	yBinCenters = np.convolve(yedges,[.5,.5],'same')
	yBinCenters = yBinCenters[1::]	

	Edges = [None]*2
	Edges[0] = xedges
	Edges[1] = yedges
 
	return hist,xBinCenters,yBinCenters,Edges
    
def loadAcel(bs,sm=2*1250):
        
        acc = MapLFPs(bs+'.acc',3).astype(float)
        accs = loadAcel_(acc,sm=sm)
                
        return accs
    
def loadACC(b,sm=2*1250):
    
        stages = LoadStages(b)
        
        acel = [None]*len(stages)
        for si,s in enumerate(stages.index):
                bs = b+'_'+str(s)
                acel[si] = loadAcel(bs,sm=2*1250)
                
        return acel
    
def estPixPerCm(tracking_,mazelen):

        x = tracking_['x'].values
        y = tracking_['y'].values

        xi = np.nanargmin(x)
        xj = np.nanargmax(x)

        yj = np.nanargmax(y)
        yi = np.nanargmin(y)

        pxmin = np.array([x[xi],y[xi]])
        pxmax = np.array([x[xj],y[xj]])
        pymax = np.array([x[yj],y[yj]])
        pymin = np.array([x[yi],y[yi]])

        mazeside = np.median([np.sum(np.sqrt(pow(pymax-pxmin,2))),\
                                np.sum(np.sqrt(pow(pymin-pxmin,2))),\
                                np.sum(np.sqrt(pow(pymax-pxmax,2))),\
                                np.sum(np.sqrt(pow(pymin-pxmax,2)))])
        pxspercm = mazeside/mazelen

        return pxspercm
    
def loaddesv(b):
        return pd.read_csv(b+'.desv',index_col=0)
    
def eeg2trki(x_,trktime,eegsr=sr):
        x_sec = x_/eegsr
        trki = np.argmin(np.abs(trktime.values-x_sec))
        return trki
    
def GetHumanReadable(size,precision=2):
        suffixes=['B','KB','MB','GB','TB']
        suffixIndex = 0
        while size > 1024 and suffixIndex < 4:
            suffixIndex += 1 #increment the index of the suffix
            size = size/1024.0 #apply the division
        return "%.*f%s"%(precision,size,suffixes[suffixIndex])
    
def getfileage(fname,unit='days'):
    
        import os.path, time
        
        age = None

        age_raw = time.time() - os.path.getmtime(fname)
        if unit == 'days':
                age = age_raw/(60*60*24)
        elif unit == 'mins':
                age = age_raw/(60*60)
        else:
                print('time unit '+unit+' not understood')
        
        return age
    
def getSessionsIs(stages,allow_stim=False,b=None):

        sbs = [stages.index[i] for (i,desen) in enumerate(stages['desen']) \
                                           if (('s b' in desen)|('sb' in desen))&(not('sbx' in desen))]
        sbs = np.where(np.in1d(stages.index,sbs))[0]
        
        aws = [stages.index[i] for (i,desen) in enumerate(stages['desen']) \
                                    if not(('s b' in desen)|('sb' in desen))]
        
        if not(allow_stim):
            
                import glob as glob
                lpulsefiles = glob.glob(b+'*.light_pulse') + glob.glob(b+'*.laser_pulse')
                
                if np.size(lpulsefiles)>0:

                        ss = stages.index
                        stms = np.zeros(len(ss),dtype=bool)
                        for si,s in enumerate(ss):
                                bs = b+'_'+str(s)
                                for suffix in ['light_pulse', 'laser_pulse']:
                                        fname = bs+'.'+suffix
                                        if os.path.exists(bs+'.'+suffix):
                                                light_ = LoadIntervals(bs,'.'+suffix).values
                                                if len(light_)>0:
                                                        stms[si] = True

                        aws = [aws_ for aws_ in aws if aws_ not in stages.index[stms]]
                
        aws = np.where(np.in1d(stages.index,aws))[0]

        return {'sb': sbs, 'aw': aws}
      
def getupos(units,trodes):

        upos = np.zeros(len(units))+np.nan

        desels = trodes.desel.values.astype('str')
        tets = trodes.index[desels=='1']
        for tet in tets:

                estpos = trodes['position'][tet]
                umask = units.trode.values==tet
                upos[umask] = estpos
                
        return upos

if False:

        import os
        import psutil

        process = psutil.Process(os.getpid())
        print(process.memory_info().rss)
