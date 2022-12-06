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

data_path= '/home/melisamc/Documentos/ripple/data/belo_pyr_output/'

figure_path = '/home/melisamc/Documentos/ripple/figures/'

rat_ID_veh = [3,4,9,201,203,206,210,211,213]
rat_ID_cbd= [2,5,10,11,204,205,207,209,212,214]

rat_number = 2
data_file = 'GC_ratID' + str(rat_ID_veh[rat_number]) + '_veh_waveforms.mat'
data = sio.loadmat(data_path + data_file)

sw_belo = data['waveforms']['sw'][0,0]['belo'][0,0]
sw_pyr = data['waveforms']['sw'][0,0]['pyr'][0,0]

r_belo = data['waveforms']['r'][0,0]['belo'][0,0]
r_pyr = data['waveforms']['r'][0,0]['pyr'][0,0]

swr_belo = data['waveforms']['swr'][0,0]['belo'][0,0]
swr_pyr = data['waveforms']['swr'][0,0]['pyr'][0,0]

cswr_belo = data['waveforms']['cswr'][0,0]['belo'][0,0]
cswr_pyr = data['waveforms']['cswr'][0,0]['pyr'][0,0]

data_file_duration = 'GC_ratID' + str(rat_ID_veh[rat_number]) + '_veh.mat'
duration = sio.loadmat(data_path + data_file_duration)

fs = 600


sf = 600
time = (np.arange(0,3601) - 3600/2 )/sf
N = int(np.sqrt(sw_belo.shape[0]))
figure, axes = plt.subplots(10, 10)
if N >= 10:
    for i in range(10):
        for j in range(10):
            index = i*10+j
            signal = sw_pyr[index, :]
            analytic_signal = hilbert(signal)
            amplitude_envelope = np.abs(analytic_signal)
            # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            # instantaneous_frequency = (np.diff(instantaneous_phase) /
            #                            (2.0 * np.pi) * fs)
            # instantaneous_frequency_filt = medfilt(instantaneous_frequency, 3)
            axes[i,j].plot(time, sw_belo[index, :], 'r', linewidth = 0.5)
            axes[i,j].plot(time, amplitude_envelope * 10, 'b', linewidth = 0.5)
            axes[i,j].set_ylim([-1500,1500])
            axes[i,j].set_xlim([-0.15,0.15])
            axes[i,j].vlines(0,-3000,3000,'k',linewidth = 0.1,linestyles='-')
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            #axes[i,j].set_aspect('equal')
plt.subplots_adjust(wspace=0, hspace=0)
figure.set_size_inches([10,10])
figure.savefig('modulation_rat_'+str(rat_ID_veh[rat_number])+'_example_SW.png')

sf = 600
time = (np.arange(0,3601) - 3600/2 )/sf
N = int(np.sqrt(r_belo.shape[0]))
figure, axes = plt.subplots(10, 10)
if N >= 10:
    for i in range(10):
        for j in range(10):
            index = i*10+j
            signal = r_pyr[index, :]
            analytic_signal = hilbert(signal)
            amplitude_envelope = np.abs(analytic_signal)
            # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            # instantaneous_frequency = (np.diff(instantaneous_phase) /
            #                            (2.0 * np.pi) * fs)
            # instantaneous_frequency_filt = medfilt(instantaneous_frequency, 3)
            axes[i,j].plot(time, r_belo[index, :], 'r', linewidth = 0.5)
            axes[i,j].plot(time, amplitude_envelope * 10, 'b', linewidth = 0.5)
            axes[i,j].set_ylim([-1500,1500])
            axes[i,j].set_xlim([-0.15,0.15])
            axes[i,j].vlines(0,-3000,3000,'k',linewidth = 0.1,linestyles='-')
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            #axes[i,j].set_aspect('equal')
plt.subplots_adjust(wspace=0, hspace=0)
figure.set_size_inches([10,10])
figure.savefig('modulation_rat_'+str(rat_ID_veh[rat_number])+'_example_Ripple.png')



sf = 600
time = (np.arange(0,3601) - 3600/2 )/sf
N = int(np.sqrt(swr_belo.shape[0]))
figure, axes = plt.subplots(10, 10)
if N >= 10:
    for i in range(10):
        for j in range(10):
            index = i*10+j
            signal = swr_pyr[index, :]
            analytic_signal = hilbert(signal)
            amplitude_envelope = np.abs(analytic_signal)
            # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            # instantaneous_frequency = (np.diff(instantaneous_phase) /
            #                            (2.0 * np.pi) * fs)
            # instantaneous_frequency_filt = medfilt(instantaneous_frequency, 3)
            axes[i,j].plot(time, swr_belo[index, :], 'r', linewidth = 0.5)
            axes[i,j].plot(time, amplitude_envelope * 10, 'b', linewidth = 0.5)
            axes[i,j].set_ylim([-1500,1500])
            axes[i,j].set_xlim([-0.15,0.15])
            axes[i,j].vlines(0,-3000,3000,'k',linewidth = 0.1,linestyles='-')
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            #axes[i,j].set_aspect('equal')
plt.subplots_adjust(wspace=0, hspace=0)
figure.set_size_inches([10,10])
figure.savefig('modulation_rat_'+str(rat_ID_veh[rat_number])+'_example_SWR.png')


sf = 600
time = (np.arange(0,3601) - 3600/2 )/sf
N = int(np.sqrt(cswr_belo.shape[0]))
figure, axes = plt.subplots(10, 10)
if N >= 10:
    for i in range(10):
        for j in range(10):
            index = i*10+j
            signal = cswr_pyr[index, :]
            analytic_signal = hilbert(signal)
            amplitude_envelope = np.abs(analytic_signal)
            axes[i,j].plot(time, cswr_belo[index, :], 'r', linewidth = 0.5)
            axes[i,j].plot(time, amplitude_envelope * 10, 'b', linewidth = 0.5)
            axes[i,j].set_ylim([-1500,1500])
            axes[i,j].set_xlim([-0.15,0.15])
            axes[i,j].vlines(0,-3000,3000,'k',linewidth = 0.1,linestyles='-')
            axes[i,j].set_xticks([])
            axes[i,j].set_yticks([])
            #axes[i,j].set_aspect('equal')
plt.subplots_adjust(wspace=0, hspace=0)
figure.set_size_inches([10,10])
figure.savefig('modulation_rat_'+str(rat_ID_veh[rat_number])+'_example_cSWR.png')

