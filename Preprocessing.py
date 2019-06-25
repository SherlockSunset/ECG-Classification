import wfdb
import matplotlib.pyplot as plt
import numpy as np
import os

"""
    Implemented by XIAOHUA, date: 25/06/19
"""

def read_ecg(file_path):
    """
    output: ecg files, get signal, annotated peaks, annotated types
    input: ecg file id
    """
    signals, fields = wfdb.rdsamp(file_path)
    annotation = wfdb.rdann(file_path, 'atr')
    # print(annotation.__dict__)
    ecg_sig = signals[:,0]
    ecg_type = annotation.symbol   ## symbol
    ecg_peak = annotation.sample  ## index

    return ecg_sig, ecg_type, ecg_peak, signals

def plot_ecg(ecg_sig, ecg_type, ecg_peak, title='Fig: Train', npeak=10, len_sig=3000):
    """
    demo plot ecg signal with annotated peaks, annotated types
    """
    _, ax = plt.subplots()
    for i in range(0, npeak):
        ax.annotate(ecg_type[i], xy=(ecg_peak[i], 0))
    ax.plot(ecg_sig[0:len_sig])
    ax.plot(ecg_peak[0:npeak], ecg_sig[ecg_peak[0:npeak]], '*')
    ax.set_title(title)
    plt.show()

def calculate_gap(ecg_peak): # feature 1: gap between successive peaks
    index_gap = []
    index_gap.append(ecg_peak[0])
    for i in range(1, len(ecg_peak)):
        gap = ecg_peak[i] - ecg_peak[i-1]
        index_gap.append(gap)
    return index_gap

def calculate_amplitude(ecg_sig, ecg_peak): #feature 2: impulse amplitude
    amplitude = []
    for index in ecg_peak:
        amplitude.append(ecg_sig[index])
    return amplitude

def assign_label(ecg_type):   ## label V and other symbols
    label_list = []
    for ele in ecg_type:
        if ele == 'V':
            label_list.append(1)
        else:
            label_list.append(-1)
    return label_list

def combine_writein(index_gap, amplitude, label_list, datadir):   # write the features into a file
    np.savetxt(r'test_'+datadir+'.txt', np.column_stack((np.array(index_gap), np.array(amplitude), np.array(label_list))), fmt='%d %.3f %d')

def combine(index_gap, amplitude):  # combine two features together
    info = []
    for (i, j) in zip(index_gap, amplitude):
        info.append([i, j])
    return info

def obtain_data_list(path):  # obtain data list
    data_list = []
    for files in os.listdir(path):
        name =os.path.splitext(files)[0]
        if name not in data_list:
            data_list.append(name)
    return data_list

if __name__ == '__main__':
    ecg_sig_data = []
    ecg_type_data = []
    ecg_peak_data = []
    data_list = obtain_data_list(path = './database/train/')
    for datadir in data_list:
        training_file_path = './database/train/' + datadir
        ecg_sig, ecg_type, ecg_peak, signals = read_ecg(training_file_path)
        index_gap_data = calculate_gap(ecg_peak)
        amplitude_data = calculate_amplitude(ecg_sig, ecg_peak)
        label_list_data = assign_label(ecg_type)
        combine_writein(index_gap_data, amplitude_data, label_list_data, datadir)
        # print(ecg_peak.shape)