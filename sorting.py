import numpy as np
import scipy

# need to implement this, basically:
# add noise timeseries + WDs
# compute welched PSD, subtract WDs that exceed SNR (to current noise+signals PSD of your threshold (typically 7))
# repeat above step at least 15-ish times


def iterative_sorting(times, hoft_full, popstrains, R, N_iter=15):
    """
    Function to perform iterative DWD subtraction by leave-one-out noise+population SNR

    Arguments
    ----------------------
    times (float array) : Array of the times at which hoft_full is sampled
    hoft_full (float array) : The full strain time-series (noise + all signals)
    df (dataframe)           : Dataframe containing the DWD population information
    popstrains (float array) : The N_dwd x time array of each binary's strain time-series

    Returns
    -----------------------
    res_idx (array)   : Current indices of the DWD df containing resolved systems.
    unres_idx (array) : Current indices of the DWD df containing unresolved systems.
    res_snrs (array)  : Final SNRs of all DWDs.

    """
    ## get welched PSD approx
    dt = times[1] - times[0]
    fs_full, fullwelch = scipy.signal.welch(
        hoft_full, fs=1 / dt, window="hann", noverlap=0.0, nperseg=256 * 8
    )

    ## bin the binaries by frequency
    ## such that the center of each bin is one of our fft frequencies
    delf = fs_full[1] - fs_full[0]
    sorted_f_idx = np.digitize(np.array(df["frequency"]), fs_full + 0.5 * delf)

    ## initialize the starting quantities
    current_hoft_full = hoft_full
    current_unres_idx = np.arange(popstrains.shape[1], dtype=int)
    current_res_idx = np.array([], dtype=int)
    resolved_snrs = np.array([], dtype=float)
    prev_N = 0

    for i in range(N_iter):
        ## make an array of fs x N_dwd
        ## where for each binary we have the fft of the time series without that binary
        ## only including systems we haven't yet resolved
        fs_i, current_welch_loo = scipy.signal.welch(
            current_hoft_full - popstrains[:, current_unres_idx].T,
            fs=1 / dt,
            window="hann",
            noverlap=0.0,
            nperseg=256 * 8,
        )
        current_latf = np.array(
            [
                current_welch_loo[ii, fidx]
                for ii, fidx in enumerate(sorted_f_idx[current_unres_idx])
            ]
        )
        ## snrs for circular, monochromatic binaries
        current_snrs = np.sqrt(
            dur_noise
            * R
            * np.array(df["amplitude"])[current_unres_idx] ** 2
            / (4 * current_latf)
        )

        ## update
        current_resolved_filt = current_snrs >= 7
        current_res_idx = np.append(
            current_res_idx, current_unres_idx[current_resolved_filt]
        )
        resolved_snrs = np.append(resolved_snrs, current_snrs[current_resolved_filt])
        current_N = len(current_res_idx)
        print("N resolved: {}".format(current_N))
        current_hoft_full = current_hoft_full - np.sum(
            popstrains[:, current_unres_idx[current_resolved_filt]], axis=1
        )
        current_unres_idx = current_unres_idx[np.invert(current_resolved_filt)]

        if current_N == prev_N:
            print(
                "Iterative subtraction has converged, stopping after {} iterations.".format(
                    i + 1
                )
            )
            break
        else:
            prev_N = current_N

    frac_res = len(current_res_idx) / popstrains.shape[1]
    print(
        "Final number of resolved binaries is {}. This is {:0.2f}% of the total catalogue.".format(
            current_N, frac_res
        )
    )

    resolved_idx, unresolved_idx, unresolved_snrs = (
        current_res_idx,
        current_unres_idx,
        current_snrs,
    )
    return resolved_idx, unresolved_idx, resolved_snrs, unresolved_snrs


def vector_sorting(binaries, fs_full, noisePSD, duration, LISA_rx, wts=1, snr_thresh=7):
    """
    Function to bin by frequency, then for the vector of binaries in each frequency bin, sort them by amplitude.

    Arguments
    -----------
    binaries (dataframe) : df with binary info. Will rephrase arguments in terms of the specific needed components later.
    fs_full (float array) : data frequencies
    noisePSD  (float)     : Level of the noise PSD in the relevant frequency bin (i.e., S_n(f))
    LISA_rx (float or array) : (pseudo) LISA response function (currently just a numerical factor)
    wts (float or array) : weights from fiducial population (1 for now)
    snr_thresh (float)    : the SNR threshold to condition resolved vs. unresolved on

    Returns
    -----------
    foreground_amp (array) : Stochastic foreground from unresolved sources, evaluated at fs_full.
    N_res (int)            : Number of resolved DWDs
    res_idx (array)        : Indices of the binaries dataframe for resolved DWDs.
    unres_idx (array)      : Indices of the binaries dataframe for unresolved DWDs.
    """
    dwd_fs = np.array(binaries["frequency"])
    dwd_amps = np.array(binaries["amplitude"])
    dwd_idx = np.arange(len(binaries["amplitude"]))
    ## bin the binaries by frequency
    ## first, find which frequency bin each binary is in
    delf = fs_full[1] - fs_full[0]
    f_idx = np.digitize(dwd_fs, fs_full + 0.5 * delf)

    ## now created a ragged list of arrays of varying sizes, corresponding to N_dwd(f_i)
    ## each entry is an array containing the indices of the DWDs in that bin, sorted by ascending amplitude*
    ##     * under the current assumption of uniform responses, this is equivalent to sorting by the naive SNR
    ##       (!! -- we will need to refine this in future)
    fbin_res_list = []
    foreground_amp = np.zeros(len(fs_full))

    for i in range(np.max(f_idx) + 1):
        fbin_mask_i = np.array(f_idx == i)
        fbin_amps_i = dwd_amps[fbin_mask_i] * np.sqrt(
            LISA_rx
        )  ## sqrt because we square the amplitudes to get Sgw
        fbin_sort_i = np.argsort(fbin_amps_i)
        re_sort_i = np.argsort(
            fbin_sort_i
        )  ## this will allow us to later return to the original order

        fbin_Nij = calc_Nij(fbin_amps_i[fbin_sort_i], noisePSD[i], wts, duration)
        res_mask_i = fbin_Nij >= snr_thresh
        res_mask_i_resort = res_mask_i[re_sort_i]
        fbin_res_list.append(dwd_idx[fbin_mask_i][res_mask_i_resort])

        foreground_amp[i] = np.sum(0.5 * fbin_amps_i[np.invert(res_mask_i_resort)] ** 2)

    ##unpack the binned list
    res_idx = np.array([], dtype=int)
    for i, arr in enumerate(fbin_res_list):
        res_idx = np.append(res_idx, arr)
    N_res = len(res_idx)
    unres_idx = np.isin(dwd_idx, res_idx, invert=True)

    return foreground_amp, N_res, res_idx, unres_idx


## from foreground.py
def calc_Nij(A, noisePSD, wts, duration):
    """
    Make the per-frequency SNR vector (dim 1xN_dwd)

    Arguments
    ------------
    A (float array)      : Sorted (ascending) DWD amplitudes
    noisePSD (float)     : Level of the noise PSD in the relevant frequency bin (i.e., S_n(f))
    wts (float or array) : weights from fiducial population (1 for now)
    """
    #     np.sqrt(dur_noise*np.array(df['amplitude'])[current_unres_idx]**2 / (4*current_latf))
    try:
        return np.sqrt(duration * A**2 / (4 * (noisePSD + np.cumsum(wts * A**2))))
    except:
        import pdb; pdb.set_trace()

#     return A / (noisePSD + np.cumsum(wts*A))
