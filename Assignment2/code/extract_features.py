import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')

def get_time_series_features(signal):
    window_size = len(signal)
    # mean
    sig_mean = np.mean(signal)
    # standard deviation
    sig_std = np.std(signal)
    # avg absolute difference
    sig_aad = np.mean(np.absolute(signal - np.mean(signal)))
    # min
    sig_min = np.min(signal)
    # max
    sig_max = np.max(signal)
    # max-min difference
    sig_maxmin_diff = sig_max - sig_min
    # median
    sig_median = np.median(signal)
    # median absolute deviation
    sig_mad = np.median(np.absolute(signal - np.median(signal)))
    # Inter-quartile range
    sig_IQR = np.percentile(signal, 75) - np.percentile(signal, 25)
    # negative count
    sig_neg_count = np.sum(s < 0 for s in signal)
    # positive count
    sig_pos_count = np.sum(s > 0 for s in signal)
    # values above mean
    sig_above_mean = np.sum(s > sig_mean for s in signal)
    # number of peaks
    sig_num_peaks = len(find_peaks(signal)[0])
    # skewness
    sig_skew = stats.skew(signal)
    # kurtosis
    sig_kurtosis = stats.kurtosis(signal)
    # energy
    sig_energy = np.sum(s ** 2 for s in signal) / window_size
    # signal area
    sig_sma = np.sum(signal) / window_size

    return [sig_mean, sig_std, sig_aad, sig_min, sig_max, sig_maxmin_diff, sig_median, sig_mad, sig_IQR, sig_neg_count, sig_pos_count, sig_above_mean, sig_num_peaks, sig_skew, sig_kurtosis, sig_energy, sig_sma]


def get_freq_domain_features(signal):
    all_fft_features = []
    window_size = len(signal)
    signal_fft = np.abs(np.fft.fft(signal))
    # Signal DC component
    sig_fft_dc = signal_fft[0]
    # aggregations over the fft signal
    fft_feats = get_time_series_features(signal_fft[1:int(window_size / 2) + 1])

    all_fft_features.append(sig_fft_dc)
    all_fft_features.extend(fft_feats)
    return all_fft_features

def get_similarity_features(signal):
    all_similarity_features = []
    window_size = len(signal)
    # autocorrelation
    sig_autocorr = np.correlate(signal, signal, mode='full')[window_size - 1:]
    # aggregations over the autocorrelation signal
    autocorr_feats = get_time_series_features(sig_autocorr)
    all_similarity_features.extend(autocorr_feats)

    # dynamic time warping
    sig_dtw = np.zeros(window_size)
    for i in range(window_size):
        sig_dtw[i] = np.sum(np.absolute(signal - np.roll(signal, i)))
    # aggregations over the dtw signal
    dtw_feats = get_time_series_features(sig_dtw)
    all_similarity_features.extend(dtw_feats)

    # euclidean distance
    sig_euclidean = np.zeros(window_size)
    for i in range(window_size):
        sig_euclidean[i] = np.sqrt(np.sum(np.square(signal - np.roll(signal, i))))
    # aggregations over the euclidean signal
    euclidean_feats = get_time_series_features(sig_euclidean)
    all_similarity_features.extend(euclidean_feats)
    return all_similarity_features

def get_wavelet_features(signal):
    all_wavelet_features = []
    window_size = len(signal)
    # wavelet transform
    sig_wavelet = np.abs(np.fft.fft(signal))
    # aggregations over the wavelet signal
    wavelet_feats = get_time_series_features(sig_wavelet)
    all_wavelet_features.extend(wavelet_feats)
    return all_wavelet_features