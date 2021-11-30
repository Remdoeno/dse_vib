'''
Note:

1. wave and wave_sliced
    a. wave: continuous wave signal, an 1-d array.
    b. wave_sliced: the sliced continuous wave using sliding window, an 2-d array, with first dimension representing time.
    c. wave can be transformed to wave_sliced with the 'sliding_window_wave' function.
    d. most functions are adapted to work with both 1-d wave and 2-d wave_sliced, however, it is recommanded to always use wave_sliced, as you can simply reshape wave to (1, n)
'''


import numpy as np
import pandas as pd
from scipy import signal

def sliding_window_wave(wave, fs, fs_slice, time_duration):
    '''
    Inputs:
    wave          (array): wave signal with shape (n,).
    fs            (float): sampling frequency.
    fs_slice      (float): how many slice is there in a second. fs must be divisible by fs_slice.
    time_duration (float): how many second for each slice.

    Outputs:
    (array): corresponding time point of output sliced wave.
    (array): modified wave signal.

    Example:
    fs = 1000
    t = np.arange(0,1,1 / fs)
    wave = np.sin(2 * np.pi * 20 * t)
    t2, wave_sliced = sliding_window_wave(wave, fs, fs_slice = 10, time_duration = 0.5)

    t2
    >>> array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ])

    wave_sliced.shape
    >>> (11, 500)
    '''

    if len(wave.shape) != 1:
        raise ValueError('wave should be a 1-d array with shape (n,)')

    if fs % fs_slice != 0:
        raise ValueError('fs must be divisible by fs_slice. Please choose another fs_slice value.')

    total_time = len(wave)/fs
    number_of_slices = int(len(wave) / fs * fs_slice) + 1
    slice_length = int(time_duration * fs)
    interval = int(fs / fs_slice)

    waves = []
    for i in range(number_of_slices):
        start = i * interval - int(slice_length / 2)
        end = start + slice_length
        if start < 0:
            slice_temp = np.concatenate((np.zeros(0 - start), wave[0:end]), axis = 0)
        elif end >= len(wave):
            slice_temp = np.concatenate((wave[start:len(wave)],np.zeros(end - len(wave))), axis = 0)
        else:
            slice_temp = wave[start:end]
        waves.append(slice_temp)

    t = np.linspace(0, total_time, number_of_slices)
    
    return t.astype(np.float32), np.array(waves, np.float32)


def filt_wave(wave, fs, lowfre, highfre):
    '''
    Inputs:
    wave          (array): wave signal with shape (n,) or wave_sliced with shape (t, n).
    fs              (int): sampling frequency.
    lowfre        (float): frequency lower bond.
    highfre       (float): frequency higher bond.

    Outputs:
    (array): filtered wave.
    '''

    kernel = signal.firwin(501, highfre / (fs / 2), pass_zero = True)
    kernel2 = signal.firwin(501, lowfre / (fs / 2), pass_zero = False)
    
    if len(wave.shape) == 1:
        wave1 = np.convolve(wave, kernel)[250:-250]
        filtwave = np.convolve(wave1, kernel2)[250:-250]
        return filtwave.astype('float32')

    elif len(wave.shape) == 2:
        filtwave_all = []
        for i in range(wave.shape[0]):
            wave1 = np.convolve(wave[i], kernel)[250:-250]
            filtwave = np.convolve(wave1, kernel2)[250:-250]
            filtwave_all.append(filtwave)
        return np.array(filtwave_all,'float32')

    else:
        raise ValueError('wave/wave_sliced should be either an 1-d or 2-d array.')


def envelop_wave(wave):
    '''
    Inputs:
    wave          (array): wave signal with shape (n,) or wave_sliced with shape (t, n).

    Outputs:
    (array): enveloped wave.
    '''

    envwave = abs(signal.hilbert(wave))
    envwave = envwave - np.mean(envwave, axis = -1, keepdims = True)

    return envwave.astype('float32')


def fft_wave(wave, fs, window = 'hann'):
    '''
    Inputs:
    wave            (array): wave signal with shape (n,) or wave_sliced with shape (t, n). When with wave_sliced, result is equal to stft.
    fs                (int): sampling frequency.
    window ('hann' or None): decide the window to use.

    Outputs:
    (array): frequency.
    (array): complex spectrum, or ft-spectrum, depending on whether it's wave or wave_sliced is input.
    '''

    if len(wave.shape) == 1:
        frequency = np.linspace(0, fs / 2, int(len(wave) / 2) + 1)
        if window == 'hann':
            window = signal.hann(len(wave))
            spec_full = np.fft.fft(wave * window) / np.mean(window) * 2 / len(wave)
        elif window == None:
            spec_full = np.fft.fft(wave) * 2 / len(wave)
        spec = spec_full[:len(frequency)]
        return frequency.astype(np.float32), spec

    elif len(wave.shape) == 2:
        frequency = np.linspace(0, fs/2, int(len(wave[0])/2)+1)
        if window == 'hann':
            window = signal.hann(len(wave[0]))
            spec_full = np.fft.fft(wave * window) / np.mean(window) * 2 / len(wave[0])
        elif window == None:
            spec_full = np.fft.fft(wave) * 2 / len(wave[0])
        spec = spec_full[:,:len(frequency)]
        return frequency.astype(np.float32), spec

    else:
        raise ValueError('wave/wave_sliced should be either an 1-d or 2-d array.')


def stft_wave(wave, fs, fs_slice, time_duration, window = 'hann'):
    '''
    Inputs:
    wave            (array): wave signal with shape (n,).
    fs                (int): sampling frequency.
    fs_slice          (int): how many slice is there in a second. fs must be divisible by fs_slice.
    time_duration   (float): how many second for each slice.
    window ('hann' or None): decide the window to use.

    Outputs:
    (array): frequency.
    (array): time
    (array): complex ft-spectrum, the result equals to result done with sliding_window_wave and fft.
    '''

    if window == None:
        window = np.zeros(shape = int(time_duration * fs)) + 1
    frequency, time, spec = signal.stft(wave,
                                        fs,
                                        window,
                                        nperseg = int(time_duration * fs),
                                        noverlap = int(time_duration * fs * (((time_duration * fs_slice) - 1) / (time_duration * fs_slice))))
    return time.astype(np.float32), frequency.astype(np.float32), spec.T * 2


def istft_spec(spec, fs, fs_slice, time_duration, window = 'hann'):
    '''
    Inputs:
    spec            (array): complex ft-spectrum.
    fs                (int): sampling frequency.
    fs_slice          (int): how many slice is there in a second. fs must be divisible by fs_slice.
    time_duration   (float): how many second for each slice.
    window ('hann' or None): decide the window to use.

    Outputs:
    (array): the continuious wave reconstructed.
    '''

    if window == None:
        window = np.zeros(shape = int(time_duration * fs)) + 1
    time, wave = signal.istft(spec.T/2,
                              fs,
                              window,
                              nperseg = int(time_duration * fs),
                              noverlap = int(time_duration * fs * (((time_duration * fs_slice) - 1)/(time_duration * fs_slice))))
    return wave.astype(np.float32)


class time_domain_features:
    '''
    Calculate common time domain features.
    '''
    def mean(self, wave):
        '''均值'''
        return np.mean(wave, axis = -1, keepdims = True)

    def absolute_mean(self, wave):
        '''绝对值均值'''
        return np.mean(abs(wave), axis = -1, keepdims = True)

    def variance(self, wave):
        '''方差'''
        return np.var(wave, axis = -1, keepdims = True)

    def standard_deviation(self, wave):
        '''标准差'''
        return np.std(wave, axis = -1, keepdims = True)

    def peak(self, wave):
        '''峰值'''
        return np.max(abs(wave), axis = -1, keepdims = True)

    def root_mean_square(self, wave):
        '''均方根'''
        return np.sqrt(np.mean(wave**2, axis = -1, keepdims = True))

    def sqrt_amplitude(self, wave):
        '''方根幅值'''
        return np.mean(np.abs(wave)**0.5, axis = -1, keepdims = True)**2

    def kurtosis(self, wave):
        '''峭度'''
        numerator = np.mean((wave - self.mean(wave))**4, axis = -1, keepdims = True)
        denominator = self.standard_deviation(wave)**4
        return numerator / denominator
        
    def skewness(self, wave):
        '''偏度'''
        numerator = np.mean((wave - self.mean(wave))**3, axis = -1, keepdims = True)
        denominator = self.standard_deviation(wave)**3
        return numerator / denominator
    
    def crest_factor(self, wave):
        '''峰值指标'''
        return self.peak(wave) / self.root_mean_square(wave)
    
    def peak2peak(self, wave):
        '''峰峰值'''
        return np.max(wave, axis = -1, keepdims = True) - np.min(wave, axis = -1, keepdims = True)
        
    def average_energy(self, wave):
        '''平均能量'''
        return np.mean(wave**2, axis = -1, keepdims = True)
    
    def margin_factor(self, wave):
        '''裕度指标'''
        return self.peak(wave) / self.sqrt_amplitude(wave)
    
    def shape_factor(self, wave):
        '''波形指标'''
        return self.root_mean_square(wave) / self.absolute_mean(wave)
    
    def impulse_factor(self, wave):
        '''脉冲指标'''
        return self.peak(wave) / self.absolute_mean(wave)

    def __call__(self, wave):
        '''输出所有，返回字典格式'''
        feature_names = ['mean', 'absolute_mean', 'variance', 'standard_deviation', 'peak', 'root_mean_square', 
            'sqrt_amplitude', 'kurtosis', 'skewness', 'crest_factor', 'peak2peak', 'average_energy', 'margin_factor',
            'shape_factor', 'impulse_factor'
        ]
        features = {}
        for f in feature_names:
            exec('features[f] = ' + 'self.' + f + '(wave).T[0]')
        return features
time_domain_features_wave = time_domain_features()


def order_amplitude_extraction_spec(t, f, tf_spec, base_frequency, orders = [1, 2, 3]):
    '''
    Inputs:
    t              (array): 1-d, corresponding time of time-frequency-spectrum.
    f              (array): 1-d, corresponding frequency of time-frequency-spectrum.
    tf_spec        (array): 2-d, the time-ferquency-spectrum.
    base_frequency (array): 1-d, the rotational speed's frequency at each time point, should have same shape with 't'.
    orders          (list): list of ints showing the orders to extract, default to [1, 2, 3]

    Outputs:
    (DataFrame): the extracted amplitude.
    '''

    tf_spec = abs(tf_spec)
    order_amplitude = {}
    for order in orders:
        order_amplitude[str(order) + 'x'] = []
        for i in range(len(base_frequency)):
            lowerbond = np.max(np.where(f - base_frequency[i] * (order - 0.1) <= 0))
            higherbond = np.min(np.where(f - base_frequency[i] * (order + 0.1) >= 0))
            order_amplitude[str(order) + 'x'].append(np.max(tf_spec[i, lowerbond: higherbond + 1]))

    return pd.DataFrame(order_amplitude, index = t)


class frequency_domain_features:
    '''
    Calculate common frequency domain features.
    '''
    def frequency_centre(self, frequency, spec):
        '''重心频率'''
        spec = spec.T
        amplitude = np.abs(spec)
        if(len(spec.shape)==2):
            frequency = np.array([frequency for _ in range(spec.shape[1])]).T
        return np.sum(amplitude*frequency,axis=0)/np.sum(amplitude,axis=0)
    
    def mean_frequency(self, frequency, spec):
        '''均值频率'''
        spec = spec.T
        if(len(spec.shape)==2):
            frequency = np.array([frequency for _ in range(spec.shape[1])]).T
        return np.mean(frequency,axis=0)
    
    def rms_frequency(self, frequency, spec):
        '''均方根频率'''
        spec = spec.T
        amplitude = np.abs(spec)
        if(len(spec.shape)==2):
            frequency = np.array([frequency for _ in range(spec.shape[1])]).T
        numerator = np.sum(amplitude*pow(frequency,2),axis=0)
        denominator = np.sum(amplitude,axis=0)
        return np.sqrt(numerator/denominator)
    
    def se_of_raw_signal(self, frequency, spec):
        '''谱熵'''
        spec = spec.T
        amplitude = np.abs(spec)
        if(len(spec.shape)==2):
            frequency = np.array([frequency for _ in range(spec.shape[1])]).T
        I = amplitude/np.sum(amplitude,axis=0)
        return -np.sum(I*np.log2(I),axis=0)

    def __call__(self, frequency, spec):
        '''输出所有，返回字典格式'''
        feature_names = ['frequency_centre', 'mean_frequency', 'rms_frequency', 'se_of_raw_signal']
        features = {}
        for f in feature_names:
            exec('features[f] = ' + 'self.' + f + '(frequency, spec)')
        return features
frequency_domain_features_spec = frequency_domain_features()