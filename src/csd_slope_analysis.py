import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from scipy import signal
from scipy.io import savemat
import pandas as pd
from scipy.signal import hilbert, chirp
from scipy.signal import medfilt
from scipy import signal
from scipy.fft import fft, ifft
from sklearn.linear_model import LinearRegression

def compute_spectrogram_regression_csd(matrix,time_range):

    fs = 600
    width = 6  # morlet2 width
    low_f = 1  # lowest frequency of interest
    high_f = 200  # highest frequency of interest
    freq = np.linspace(low_f, high_f, num=int(high_f / low_f))  # frequency resolution of time-frequency plot
    freq_log = np.log10(freq)
    widths = width * fs / (2 * freq * np.pi)  # wavelet widths for all frequencies of interest

    vector_intercept= np.zeros_like(matrix)
    vector_coef = np.zeros_like(matrix)
    freq_index1 = np.where(freq>1)[0]
    freq_index2 = np.where(freq<1000)[0]
    freq_index = np.intersect1d(freq_index1, freq_index2)
    for index in range(matrix.shape[2]):
        for channel in range(matrix.shape[1]):
            data_signal = matrix[:,channel,index]
            cwtm = signal.cwt(data_signal, signal.morlet2, widths, w=width)

            for t in range(1800 - time_range, 1800 + time_range):
                x = np.log10(np.abs(cwtm[:,t]))
                reg= LinearRegression().fit(freq_log[freq_index].reshape(-1,1), x[freq_index].reshape(-1,1))
                vector_coef[t,channel,index] = reg.coef_[0]
                vector_intercept[t,channel,index] = reg.intercept_[0]

    return vector_intercept,vector_coef

data_path= '/home/melisamc/Documentos/ripple/data/csd/'
figure_path = '/home/melisamc/Documentos/ripple/figures/'
files_name = ['SW','R','SWR','cSWR']
colors = ['r','b','purple','orange']
dict_name = ['csd_mat_SW','csd_mat_Ripp','csd_mat_SWR','csd_mat_cSWR']
fs = 600

type = 2
data = sio.loadmat(data_path + files_name[type] + '.mat')
matrix = data[dict_name[type]]

time_range = 150
[offset_SW,slope_SW] = compute_spectrogram_regression_csd(matrix,time_range)
mean_slope_SW = np.mean(slope_SW,axis = 2)
std_slope_SW = np.mean(slope_SW,axis = 2) / np.sqrt(slope_SW.shape[2])
mean_offset_SW = np.mean(offset_SW,axis = 2)
std_offset_SW = np.mean(offset_SW,axis = 2) / np.sqrt(offset_SW.shape[2])


figure, axes = plt.subplots(1,2)
time_axes = np.arange(0,3601)/fs - 3
for i in range(mean_slope_SW.shape[1]):
    slope = (mean_slope_SW[:,i] - np.min(mean_slope_SW[:,i])) / (np.max(mean_slope_SW[:,i]) - np.min(mean_slope_SW[:,i]))
    slope = mean_slope_SW[:, i]
    axes[0].plot(time_axes,slope + i, c =colors[type])
    offset = (mean_offset_SW[:,i] - np.min(mean_offset_SW[:,i])) / (np.max(mean_offset_SW[:,i]) - np.min(mean_offset_SW[:,i]))
    offset = mean_offset_SW[:, i]
    axes[1].plot(time_axes,offset + i, c = colors[type])

for ax in range(2):
    axes[ax].set_xlim([-0.25,0.24])
    axes[ax].set_yticks([])
    axes[ax].set_xlabel('Time (s)', fontsize = 15)
axes[0].set_title('Slope', fontsize = 15)
axes[1].set_title('Offset', fontsize = 15)

figure.set_size_inches([6,6])
figure.savefig(figure_path + 'csd_slope_' + files_name[type] + '.png' )
np.save(data_path + 'csd_SWR_slope.npy',slope_SW)
np.save(data_path + 'csd_SWR_offset.npy',offset_SW)

type = 3
data = sio.loadmat(data_path + files_name[type] + '.mat')
matrix = data[dict_name[type]]

time_range = 150
[offset_SW,slope_SW] = compute_spectrogram_regression_csd(matrix,time_range)
mean_slope_SW = np.mean(slope_SW,axis = 2)
std_slope_SW = np.mean(slope_SW,axis = 2) / np.sqrt(slope_SW.shape[2])
mean_offset_SW = np.mean(offset_SW,axis = 2)
std_offset_SW = np.mean(offset_SW,axis = 2) / np.sqrt(offset_SW.shape[2])


figure, axes = plt.subplots(1,2)
time_axes = np.arange(0,3601)/fs - 3
for i in range(mean_slope_SW.shape[1]):
    slope = (mean_slope_SW[:,i] - np.min(mean_slope_SW[:,i])) / (np.max(mean_slope_SW[:,i]) - np.min(mean_slope_SW[:,i]))
    slope = mean_slope_SW[:, i]
    axes[0].plot(time_axes,slope + i, c =colors[type])
    offset = (mean_offset_SW[:,i] - np.min(mean_offset_SW[:,i])) / (np.max(mean_offset_SW[:,i]) - np.min(mean_offset_SW[:,i]))
    offset = mean_offset_SW[:, i]
    axes[1].plot(time_axes,offset + i, c = colors[type])

for ax in range(2):
    axes[ax].set_xlim([-0.25,0.24])
    axes[ax].set_yticks([])
    axes[ax].set_xlabel('Time (s)', fontsize = 15)
axes[0].set_title('Slope', fontsize = 15)
axes[1].set_title('Offset', fontsize = 15)

figure.set_size_inches([6,6])
figure.savefig(figure_path + 'csd_slope_' + files_name[type] + '.png' )
np.save(data_path + 'csd_cSWR_slope.npy',slope_SW)
np.save(data_path + 'csd_cSWR_offset.npy',offset_SW)

type = 0
data = sio.loadmat(data_path + files_name[type] + '.mat')
matrix = data[dict_name[type]]

time_range = 150
[offset_SW,slope_SW] = compute_spectrogram_regression_csd(matrix,time_range)
mean_slope_SW = np.mean(slope_SW,axis = 2)
std_slope_SW = np.mean(slope_SW,axis = 2) / np.sqrt(slope_SW.shape[2])
mean_offset_SW = np.mean(offset_SW,axis = 2)
std_offset_SW = np.mean(offset_SW,axis = 2) / np.sqrt(offset_SW.shape[2])


figure, axes = plt.subplots(1,2)
time_axes = np.arange(0,3601)/fs - 3
for i in range(mean_slope_SW.shape[1]):
    slope = (mean_slope_SW[:,i] - np.min(mean_slope_SW[:,i])) / (np.max(mean_slope_SW[:,i]) - np.min(mean_slope_SW[:,i]))
    slope = mean_slope_SW[:, i]
    axes[0].plot(time_axes,slope + i, c =colors[type])
    offset = (mean_offset_SW[:,i] - np.min(mean_offset_SW[:,i])) / (np.max(mean_offset_SW[:,i]) - np.min(mean_offset_SW[:,i]))
    offset = mean_offset_SW[:, i]
    axes[1].plot(time_axes,offset + i, c = colors[type])

for ax in range(2):
    axes[ax].set_xlim([-0.25,0.24])
    axes[ax].set_yticks([])
    axes[ax].set_xlabel('Time (s)', fontsize = 15)
axes[0].set_title('Slope', fontsize = 15)
axes[1].set_title('Offset', fontsize = 15)

figure.set_size_inches([6,6])
figure.savefig(figure_path + 'csd_slope_' + files_name[type] + '.png' )
np.save(data_path + 'csd_SW_slope.npy',slope_SW)
np.save(data_path + 'csd_SW_offset.npy',offset_SW)


slope_SW = np.load(data_path + 'csd_SW_slope.npy')
offset_SW = np.load(data_path + 'csd_SW_offset.npy')
slope_R = np.load(data_path + 'csd_R_slope.npy')
offset_R = np.load(data_path + 'csd_R_offset.npy')
slope_SWR = np.load(data_path + 'csd_SWR_slope.npy')
offset_SWR = np.load(data_path + 'csd_SWR_offset.npy')
slope_cSWR = np.load(data_path + 'csd_cSWR_slope.npy')
offset_cSWR = np.load(data_path + 'csd_cSWR_offset.npy')

slope_list = []
slope_list.append(slope_SW)
slope_list.append(slope_R)
slope_list.append(slope_SWR)
slope_list.append(slope_cSWR)

offset_list = []
offset_list.append(offset_SW)
offset_list.append(offset_R)
offset_list.append(offset_SWR)
offset_list.append(offset_cSWR)


figure, axes = plt.subplots(1,2)
time_axes = np.arange(0,3601)/fs - 3
for j in range(4):
    slope = np.mean(slope_list[j], axis=2)
    offset = np.mean(offset_list[j], axis=2)
    for i in range(slope.shape[1]):
        axes[0].plot(time_axes,slope[:,i] + i, c =colors[j], linewidth = 1)
        axes[1].plot(time_axes,offset[:,i] + i, c = colors[j], linewidth = 1)

for ax in range(2):
    axes[ax].set_xlim([-0.1,0.1])
    axes[ax].set_yticks([])
    axes[ax].set_xlabel('Time (s)', fontsize = 15)
axes[0].set_title('Slope', fontsize = 15)
axes[1].set_title('Offset', fontsize = 15)

figure.set_size_inches([6,6])
figure.savefig(figure_path + 'csd_slope.png' )

from matplotlib import colors
cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)


COLOR = (0.0, 0.1, 0)

def color_conv(color_range):
    return (COLOR[0] + color_range, COLOR[1], COLOR[2])

figure, axes = plt.subplots(2,4)
time_axes = np.arange(0,3601)/fs - 3
for j in range(4):
    slope = np.mean(slope_list[j], axis=2)
    offset = np.mean(offset_list[j], axis=2)
    for i in range(slope.shape[1]):
        axes[0,j].plot(time_axes,slope[:,i], color=color_conv(i/30), linewidth = 1)
        axes[1,j].plot(time_axes,offset[:,i], color=color_conv(i/30), linewidth = 1)

for j in range(4):
    for ax in range(2):
        axes[ax,j].set_xlim([-0.25,0.24])
        axes[ax,j].set_xlabel('Time (s)', fontsize = 15)
    axes[0,j].set_title('Slope', fontsize = 15)
    axes[1,j].set_title('Offset', fontsize = 15)
    axes[0,j].set_ylim([-3,-1])
    axes[1,j].set_ylim([0,6])


figure.set_size_inches([15,10])
figure.savefig(figure_path + 'csd_slope_3.png' )


figure, axes = plt.subplots(2,4)
time_axes = np.arange(0,3601)/fs - 3
for j in range(4):
    slope = np.mean(slope_list[j], axis=2)
    offset = np.mean(offset_list[j], axis=2)
    for i in range(slope.shape[1]):
        axes[0,j].plot(time_axes,slope[:,i]+i*0.1, color=color_conv(i/30), linewidth = 1)
        axes[1,j].plot(time_axes,offset[:,i]+i*0.1, color=color_conv(i/30), linewidth = 1)

for j in range(4):
    for ax in range(2):
        axes[ax,j].set_xlim([-0.25,0.24])
        axes[ax,j].set_xlabel('Time (s)', fontsize = 15)
    axes[0,j].set_title('Slope', fontsize = 15)
    axes[1,j].set_title('Offset', fontsize = 15)
    axes[0,j].set_yticks([])
    axes[1,j].set_yticks([])

    #axes[0,j].set_ylim([-3,-1])
    #axes[1,j].set_ylim([0,6])


figure.set_size_inches([15,10])
figure.savefig(figure_path + 'csd_slope_3_.png' )



figure, axes = plt.subplots(1,4)
time_axes = np.arange(0,3601)/fs - 3
for j in range(4):
    offset = np.mean(offset_list[j], axis=2)
    for i in range(slope.shape[1]):
        axes[j].plot(time_axes,offset[:,i]+i*0.1, color=color_conv(i/30), linewidth = 1)

for j in range(4):
    axes[j].set_xlim([-0.25,0.24])
    axes[j].set_xlabel('Time (s)', fontsize = 15)
    axes[j].set_title('Offset: ' + files_name[j] , fontsize = 15)
    axes[j].set_yticks([])
    #axes[0,j].set_ylim([-3,-1])
    #axes[1,j].set_ylim([0,6])

figure.set_size_inches([15,6])
figure.savefig(figure_path + 'csd_offset_only_.png' )




figure, axes = plt.subplots(1,4)
time_axes = np.arange(0,3601)/fs - 3
for j in range(4):
    slope = np.mean(slope_list[j], axis=2)
    for i in range(slope.shape[1]):
        axes[j].plot(time_axes,slope[:,i]+i*0.1, color=color_conv(i/30), linewidth = 1)

for j in range(4):
    axes[j].set_xlim([-0.25,0.24])
    axes[j].set_xlabel('Time (s)', fontsize = 15)
    axes[j].set_title('Slope:' + files_name[j] , fontsize = 15)
    axes[j].set_yticks([])
    #axes[0,j].set_ylim([-3,-1])
    #axes[1,j].set_ylim([0,6])

figure.set_size_inches([15,6])
figure.savefig(figure_path + 'csd_slope_only_.png' )

##########################

type = 0
data = sio.loadmat(data_path + files_name[type] + '.mat')
matrix_SW = data[dict_name[type]]
type = 1
data = sio.loadmat(data_path + files_name[type] + '.mat')
matrix_R = data[dict_name[type]]
type = 2
data = sio.loadmat(data_path + files_name[type] + '.mat')
matrix_SWR = data[dict_name[type]]
type = 3
data = sio.loadmat(data_path + files_name[type] + '.mat')
matrix_cSWR = data[dict_name[type]]

matrix_list = []
matrix_list.append(matrix_SW)
matrix_list.append(matrix_R)
matrix_list.append(matrix_SWR)
matrix_list.append(matrix_cSWR)


from matplotlib import colors
cdict = {'red':   ((0.0,  0.22, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.89, 1.0)),

         'green': ((0.0,  0.49, 0.0),
                   (0.5,  1.0, 1.0),
                   (1.0,  0.12, 1.0)),

         'blue':  ((0.0,  0.72, 0.0),
                   (0.5,  0.0, 0.0),
                   (1.0,  0.11, 1.0))}

cmap = colors.LinearSegmentedColormap('custom', cdict)
COLOR = (0.0, 0.1, 0)
def color_conv(color_range):
    return (COLOR[0] + color_range, COLOR[1], COLOR[2])

figure, axes = plt.subplots(1,4)
time_axes = np.arange(0,3601)/fs - 3
for j in range(4):
    slope = np.mean(matrix_list[j], axis=2)
    for i in range(slope.shape[1]):
        axes[j].plot(time_axes,slope[:,i], color=color_conv(i/30), linewidth = 1)

for j in range(4):
    for ax in range(2):
        axes[j].set_xlim([-0.25,0.24])
        axes[j].set_xlabel('Time (s)', fontsize = 15)
    axes[j].set_title('CSD', fontsize = 15)
    axes[j].set_ylim([-80,80])


figure.set_size_inches([15,6])
figure.savefig(figure_path + 'csd.png' )



figure, axes = plt.subplots(1,4)
time_axes = np.arange(0,3601)/fs - 3
for j in range(4):
    slope = np.mean(matrix_list[j], axis=2)
    for i in range(slope.shape[1]):
        axes[j].plot(time_axes,np.abs(slope[:,i]), color=color_conv(i/30), linewidth = 1)

for j in range(4):
    for ax in range(2):
        axes[j].set_xlim([-0.25,0.24])
        axes[j].set_xlabel('Time (s)', fontsize = 15)
    axes[j].set_title('CSD', fontsize = 15)
    axes[j].set_ylim([0,80])


figure.set_size_inches([15,6])
figure.savefig(figure_path + 'abs_csd.png' )



figure, axes = plt.subplots(1,4)
time_axes = np.arange(0,3601)/fs - 3
for j in range(4):
    slope = np.mean(matrix_list[j], axis=2)
    for i in range(slope.shape[1]):
        axes[j].plot(time_axes,slope[:,i]+10*i, color=color_conv(i/30), linewidth = 1)

for j in range(4):
    axes[j].set_xlim([-0.25,0.24])
    axes[j].set_xlabel('Time (s)', fontsize = 15)
    axes[j].set_title('CSD ' + files_name[j], fontsize = 15)
    #axes[j].set_ylim([0,80])
    axes[j].set_yticks([])

figure.set_size_inches([15,6])
figure.savefig(figure_path + 'csd_2.png' )