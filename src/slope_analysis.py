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

def compute_spectrogram_regression(swr_belo,time_range):

    vector_intercept_belo = np.zeros((swr_belo.shape[0],3600))
    vector_coef_belo = np.zeros((swr_belo.shape[0],3600))

    for index in range(swr_belo.shape[0]):
        data_signal = swr_belo[index, :]
        cwtm = signal.cwt(data_signal, signal.morlet2, widths, w=width)

        for t in range(1800 - time_range, 1800 + time_range):
            x = np.log10(np.abs(cwtm[:,t]))
            reg= LinearRegression().fit(freq_log.reshape(-1,1), x.reshape(-1,1))
            vector_coef_belo[index,t] = reg.coef_[0]
            vector_intercept_belo[index,t] = reg.intercept_[0]

    return vector_intercept_belo,vector_coef_belo



data_path= '/home/melisamc/Documentos/ripple/data/belo_pyr_raw_output/'
data_path_output = '/home/melisamc/Documentos/ripple/data/features_meli/'

figure_path = '/home/melisamc/Documentos/ripple/figures/'

rat_ID_veh = [3,4,9,201,203,206,210,211,213]
rat_ID_cbd= [2,5,10,11,204,205,207,209,212,214]

fs = 600
width = 6  # morlet2 width
low_f = 1  # lowest frequency of interest
high_f = 200  # highest frequency of interest
freq = np.linspace(low_f, high_f, num=int(high_f / low_f))  # frequency resolution of time-frequency plot
freq_log = np.log10(freq)
widths = width * fs / (2 * freq * np.pi)  # wavelet widths for all frequencies of interest

for rat_number in range(len(rat_ID_veh)):
    data_file = 'GC_ratID' + str(rat_ID_veh[rat_number]) + '_veh_waveforms.mat'
    data = sio.loadmat(data_path + data_file)

    sw_belo = data['waveforms']['sw'][0,0]['belo_ray'][0,0]
    sw_pyr = data['waveforms']['sw'][0,0]['pyr_raw'][0,0]
    r_belo = data['waveforms']['r'][0,0]['belo_ray'][0,0]
    r_pyr = data['waveforms']['r'][0,0]['pyr_raw'][0,0]
    swr_belo = data['waveforms']['swr'][0,0]['belo_ray'][0,0]
    swr_pyr = data['waveforms']['swr'][0,0]['pyr_raw'][0,0]
    cswr_belo = data['waveforms']['cswr'][0,0]['belo_ray'][0,0]
    cswr_pyr = data['waveforms']['cswr'][0,0]['pyr_raw'][0,0]

    data_file_duration = 'FT_ratID' + str(rat_ID_veh[rat_number]) + '_veh.mat'
    duration = sio.loadmat(data_path + data_file_duration)
    duration_sw= np.diff(duration['duration']['sw'][0, 0], axis=1)
    duration_r= np.diff(duration['duration']['r'][0, 0], axis=1)
    duration_swr= np.diff(duration['duration']['swr'][0, 0], axis=1)
    duration_cswr= np.diff(duration['duration']['cswr'][0, 0], axis=1)

    sw_intercept_belo, sw_coef_belo = compute_spectrogram_regression(sw_belo,300)
    sw_intercept_pyr, sw_coef_pyr = compute_spectrogram_regression(sw_pyr,300)
    r_intercept_belo, r_coef_belo = compute_spectrogram_regression(r_belo,300)
    r_intercept_pyr, r_coef_pyr = compute_spectrogram_regression(r_pyr,300)
    swr_intercept_belo, swr_coef_belo = compute_spectrogram_regression(swr_belo,300)
    swr_intercept_pyr, swr_coef_pyr = compute_spectrogram_regression(swr_pyr,300)
    cswr_intercept_belo, cswr_coef_belo = compute_spectrogram_regression(cswr_belo,300)
    cswr_intercept_pyr, cswr_coef_pyr = compute_spectrogram_regression(cswr_pyr,300)





# figure, axes = plt.subplots(1,2)
# axes[0].plot(np.abs(cwtm[:,1800+300]),c = 'k')
# axes[0].plot(np.abs(cwtm[:,1800]),'purple')
# axes[0].set_xlim([0,200])
# axes[0].set_xlabel('Frequency [Hz]')
# axes[0].set_ylabel('Power')
# axes[0].legend(['PostEvent','Event'], fontsize = 10)
# axes[1].plot(np.abs(cwtm[:,1800+300]),c = 'k')
# axes[1].plot(np.abs(cwtm[:,1800]),'purple')
# x = np.log10(np.abs(cwtm[:, 1800+300]))
# reg = LinearRegression().fit(freq_log.reshape(-1, 1), x.reshape(-1, 1))
# y = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
# axes[1].plot(freq,y,'grey')
# x = np.log10(np.abs(cwtm[:, 1800]))
# reg = LinearRegression().fit(freq_log.reshape(-1, 1), x.reshape(-1, 1))
# y = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
# axes[1].plot(freq,y,'violet')
# axes[1].set_xscale('log')
# axes[1].set_yscale('log')
# axes[1].set_xlabel('Frequency [Hz]')
# figure.suptitle('SWR Radiatum',fontsize =12)
# figure.set_size_inches([8,4])
# figure.savefig(figure_path + 'slope_SWR_rat_'+str(rat_ID_veh[rat_number])+'_example_rad_regresion.png')


# time = np.arange(0,3600)/fs - 3
# select = np.arange(1800-300,1800+300)
# figure, axes = plt.subplots(2,2)
# for i in range(swr_coef_belo.shape[0]):
#     axes[0,0].plot(time[select],sw_coef_belo[i,select],c = 'r',alpha = 0.05)
# axes[0,0].set_ylabel('Slope', fontsize = 12)
# axes[0,0].set_xlabel('Time', fontsize = 12)
# axes[0,0].set_title('Slope SWR Radiatum', fontsize = 15)
# axes[0,0].set_ylim([-2.5,0.5])
#
# for i in range(swr_coef_belo.shape[0]):
#     axes[0,1].plot(time[select],sw_intercept_belo[i,select],c = 'r',alpha = 0.05)
# axes[0,1].set_ylabel('Intercept', fontsize = 12)
# axes[0,1].set_xlabel('Time', fontsize = 12)
# axes[0,1].set_title('Intercept SWR Radiatum', fontsize = 15)
# axes[0,1].set_ylim([1,6])
#
# for i in range(swr_coef_belo.shape[0]):
#     axes[1,0].plot(time[select],sw_coef_pyr[i,select],c = 'b',alpha = 0.05)
# axes[1,0].set_ylabel('Slope', fontsize = 12)
# axes[1,0].set_xlabel('Time', fontsize = 12)
# axes[1,0].set_title('Slope SWR Pyramidale', fontsize = 15)
# axes[1,0].set_ylim([-2.5,0.5])
#
# for i in range(swr_coef_belo.shape[0]):
#     axes[1,1].plot(time[select],sw_intercept_pyr[i,select],c = 'b',alpha = 0.05)
# axes[1,1].set_ylabel('Intercept', fontsize = 12)
# axes[1,1].set_xlabel('Time', fontsize = 12)
# axes[1,1].set_title('Intercept SWR Pyramidale', fontsize = 15)
# axes[1,1].set_ylim([2,6])
#
# figure.set_size_inches([10,10])
# figure.savefig(figure_path + 'slope_SW_rat_'+str(rat_ID_veh[rat_number])+'.png')
#

time = np.arange(0,3600)/fs - 3
select = np.arange(1800-300,1800+300)

figure, axes = plt.subplots(2,4)
for i in range(swr_coef_belo.shape[0]):
    axes[0,0].plot(time[select],sw_coef_belo[i,select],c = 'r',alpha = 0.05)
    axes[1,0].plot(time[select],sw_coef_pyr[i,select],c = 'k',alpha = 0.05)
axes[0,0].set_ylabel('Slope', fontsize = 12)
axes[0,0].set_xlabel('Time', fontsize = 12)
axes[0,0].set_title('SW Radiatum', fontsize = 15)
axes[0,0].set_ylim([-2.5,0.5])
axes[1,0].set_ylabel('Slope', fontsize = 12)
axes[1,0].set_xlabel('Time', fontsize = 12)
axes[1,0].set_title('SW Pyramidale', fontsize = 15)
axes[1,0].set_ylim([-2.5,0.5])

for i in range(r_coef_belo.shape[0]):
    axes[0,1].plot(time[select],r_coef_belo[i,select],c = 'k',alpha = 0.05)
    axes[1,1].plot(time[select],r_coef_pyr[i,select],c = 'b',alpha = 0.05)
axes[0,1].set_ylabel('Slope', fontsize = 12)
axes[0,1].set_xlabel('Time', fontsize = 12)
axes[0,1].set_title('R Radiatum', fontsize = 15)
axes[0,1].set_ylim([-2.5,0.5])
axes[1,1].set_ylabel('Slope', fontsize = 12)
axes[1,1].set_xlabel('Time', fontsize = 12)
axes[1,1].set_title('R Pyramidale', fontsize = 15)
axes[1,1].set_ylim([-2.5,0.5])

for i in range(swr_coef_belo.shape[0]):
    axes[0,2].plot(time[select],swr_coef_belo[i,select],c = 'purple',alpha = 0.05)
    axes[1,2].plot(time[select],swr_coef_pyr[i,select],c = 'purple',alpha = 0.05)
axes[0,2].set_ylabel('Slope', fontsize = 12)
axes[0,2].set_xlabel('Time', fontsize = 12)
axes[0,2].set_title('SWR Radiatum', fontsize = 15)
axes[0,2].set_ylim([-2.5,0.5])
axes[1,2].set_ylabel('Slope', fontsize = 12)
axes[1,2].set_xlabel('Time', fontsize = 12)
axes[1,2].set_title('SWR Pyramidale', fontsize = 15)
axes[1,2].set_ylim([-2.5,0.5])


for i in range(cswr_coef_belo.shape[0]):
    axes[0,3].plot(time[select],cswr_coef_belo[i,select],c = 'orange',alpha = 0.05)
    axes[1,3].plot(time[select],cswr_coef_pyr[i,select],c = 'orange',alpha = 0.05)
axes[0,3].set_ylabel('Slope', fontsize = 12)
axes[0,3].set_xlabel('Time', fontsize = 12)
axes[0,3].set_title('cSWR Radiatum', fontsize = 15)
axes[0,3].set_ylim([-2.5,0.5])
axes[1,3].set_ylabel('Slope', fontsize = 12)
axes[1,3].set_xlabel('Time', fontsize = 12)
axes[1,3].set_title('cSWR Pyramidale', fontsize = 15)
axes[1,3].set_ylim([-2.5,0.5])

figure.set_size_inches([20,10])
figure.savefig(figure_path + 'slope_rat_'+str(rat_ID_veh[rat_number])+'.png')

##########################################################################################3
figure, axes = plt.subplots(1,4)
axes[0].plot(time[select],np.mean(sw_coef_belo[:,select],axis=0),c = 'r',alpha = 0.8)
axes[0].plot(time[select],np.mean(sw_coef_pyr[:,select],axis=0),c = 'k',alpha = 0.8)
axes[0].set_ylabel('Slope', fontsize = 12)
axes[0].set_xlabel('Time', fontsize = 12)
axes[0].set_title('SW', fontsize = 15)
axes[0].set_ylim([-2.5,0.5])
axes[0].legend(['Radiatum','Piramidale'],fontsize = 12)

axes[1].plot(time[select],np.mean(r_coef_belo[:,select],axis=0),c = 'k',alpha = 0.8)
axes[1].plot(time[select],np.mean(r_coef_pyr[:,select],axis=0),c = 'b',alpha = 0.8)
axes[1].set_ylabel('Slope', fontsize = 12)
axes[1].set_xlabel('Time', fontsize = 12)
axes[1].set_title('R Radiatum', fontsize = 15)
axes[1].set_ylim([-2.5,0.5])
axes[1].legend(['Radiatum','Piramidale'],fontsize = 12)

axes[2].plot(time[select],np.mean(swr_coef_belo[:,select],axis=0),c = 'darkviolet',alpha = 0.8)
axes[2].plot(time[select],np.mean(swr_coef_pyr[:,select],axis=0),c = 'violet',alpha = 0.8)
axes[2].set_ylabel('Slope', fontsize = 12)
axes[2].set_xlabel('Time', fontsize = 12)
axes[2].set_title('SWR Radiatum', fontsize = 15)
axes[2].set_ylim([-2.5,0.5])
axes[2].legend(['Radiatum','Piramidale'],fontsize = 12)

axes[3].plot(time[select],np.mean(cswr_coef_belo[:,select],axis = 0),c = 'darkorange',alpha = 0.8)
axes[3].plot(time[select],np.mean(cswr_coef_pyr[:,select],axis=0),c = 'orange',alpha = 0.8)
axes[3].set_ylabel('Slope', fontsize = 12)
axes[3].set_xlabel('Time', fontsize = 12)
axes[3].set_title('cSWR Radiatum', fontsize = 15)
axes[3].set_ylim([-2.5,0.5])
axes[3].legend(['Radiatum','Piramidale'],fontsize = 12)

figure.set_size_inches([20,5])
figure.savefig(figure_path + 'slope_mean_rat_'+str(rat_ID_veh[rat_number])+'.png')
####################################################################33

figure, axes = plt.subplots(1,4)
axes[0].plot(time[select],np.mean(sw_intercept_belo[:,select],axis=0),c = 'r',alpha = 0.8)
axes[0].plot(time[select],np.mean(sw_intercept_pyr[:,select],axis=0),c = 'k',alpha = 0.8)
axes[0].set_ylabel('Offset', fontsize = 12)
axes[0].set_xlabel('Time', fontsize = 12)
axes[0].set_title('SW', fontsize = 15)
axes[0].set_ylim([2,6])
axes[0].legend(['Radiatum','Piramidale'],fontsize = 12)

axes[1].plot(time[select],np.mean(r_intercept_belo[:,select],axis=0),c = 'k',alpha = 0.8)
axes[1].plot(time[select],np.mean(r_intercept_pyr[:,select],axis=0),c = 'b',alpha = 0.8)
axes[1].set_ylabel('Offset', fontsize = 12)
axes[1].set_xlabel('Time', fontsize = 12)
axes[1].set_title('R Radiatum', fontsize = 15)
axes[1].set_ylim([2,6])
axes[1].legend(['Radiatum','Piramidale'],fontsize = 12)

axes[2].plot(time[select],np.mean(swr_intercept_belo[:,select],axis=0),c = 'darkviolet',alpha = 0.8)
axes[2].plot(time[select],np.mean(swr_intercept_pyr[:,select],axis=0),c = 'violet',alpha = 0.8)
axes[2].set_ylabel('Offset', fontsize = 12)
axes[2].set_xlabel('Time', fontsize = 12)
axes[2].set_title('SWR Radiatum', fontsize = 15)
axes[2].set_ylim([2,6])
axes[2].legend(['Radiatum','Piramidale'],fontsize = 12)

axes[3].plot(time[select],np.mean(cswr_intercept_belo[:,select],axis = 0),c = 'darkorange',alpha = 0.8)
axes[3].plot(time[select],np.mean(cswr_intercept_pyr[:,select],axis=0),c = 'orange',alpha = 0.8)
axes[3].set_ylabel('Offset', fontsize = 12)
axes[3].set_xlabel('Time', fontsize = 12)
axes[3].set_title('cSWR Radiatum', fontsize = 15)
axes[3].set_ylim([2,6])
axes[3].legend(['Radiatum','Piramidale'],fontsize = 12)

figure.set_size_inches([20,5])
figure.savefig(figure_path + 'offset_mean_rat_'+str(rat_ID_veh[rat_number])+'.png')

########################################################################################

time = np.arange(0,3600)/fs - 3
select = np.arange(1800-300,1800+300)

def create_time_histogram(sw_coef_belo):
    bins = np.arange(-3,0,0.1)
    histo_output = np.zeros((len(bins)-1,sw_coef_belo.shape[1]))
    for t in range(sw_coef_belo.shape[1]):
        x = np.histogram(sw_coef_belo[:,t],bins = bins)
        histo_output[:,t] = x[0]
    return histo_output

sw_coef_belo_histo = create_time_histogram(sw_coef_belo)
sw_coef_pyr_histo = create_time_histogram(sw_coef_pyr)
r_coef_belo_histo = create_time_histogram(r_coef_belo)
r_coef_pyr_histo = create_time_histogram(r_coef_pyr)
swr_coef_belo_histo = create_time_histogram(swr_coef_belo)
swr_coef_pyr_histo = create_time_histogram(swr_coef_pyr)
cswr_coef_belo_histo = create_time_histogram(cswr_coef_belo)
cswr_coef_pyr_histo = create_time_histogram(cswr_coef_pyr)

bins = np.arange(-3, -0.1, 0.1)

figure, axes = plt.subplots(2,4)
axes[0,0].pcolormesh(time[select],bins,sw_coef_belo_histo[:,select],cmap = 'Reds')
axes[1,0].pcolormesh(time[select],bins,sw_coef_pyr_histo[:,select],cmap = 'Greys',)
axes[0,0].set_ylabel('Slope', fontsize = 12)
axes[0,0].set_xlabel('Time', fontsize = 12)
axes[0,0].set_title('SW Radiatum', fontsize = 15)
axes[1,0].set_ylabel('Slope', fontsize = 12)
axes[1,0].set_xlabel('Time', fontsize = 12)
axes[1,0].set_title('SW Pyramidale', fontsize = 15)

axes[0, 1].pcolormesh(time[select], bins, r_coef_belo_histo[:, select], cmap='Greys')
axes[1, 1].pcolormesh(time[select], bins, r_coef_pyr_histo[:, select], cmap='Blues' )
axes[0,1].set_ylabel('Slope', fontsize = 12)
axes[0,1].set_xlabel('Time', fontsize = 12)
axes[0,1].set_title('R Radiatum', fontsize = 15)
axes[1,1].set_ylabel('Slope', fontsize = 12)
axes[1,1].set_xlabel('Time', fontsize = 12)
axes[1,1].set_title('R Pyramidale', fontsize = 15)

axes[0, 2].pcolormesh(time[select], bins, swr_coef_belo_histo[:, select], cmap='Purples')
axes[1, 2].pcolormesh(time[select], bins, swr_coef_pyr_histo[:, select], cmap='BuPu')
axes[0,2].set_ylabel('Slope', fontsize = 12)
axes[0,2].set_xlabel('Time', fontsize = 12)
axes[0,2].set_title('SWR Radiatum', fontsize = 15)
axes[1,2].set_ylabel('Slope', fontsize = 12)
axes[1,2].set_xlabel('Time', fontsize = 12)
axes[1,2].set_title('SWR Pyramidale', fontsize = 15)


axes[0, 3].pcolormesh(time[select], bins, cswr_coef_belo_histo[:, select], cmap='OrRd')
axes[1, 3].pcolormesh(time[select], bins, cswr_coef_pyr_histo[:, select], cmap='Oranges')
axes[0,3].set_ylabel('Slope', fontsize = 12)
axes[0,3].set_xlabel('Time', fontsize = 12)
axes[0,3].set_title('cSWR Radiatum', fontsize = 15)
axes[1,3].set_ylabel('Slope', fontsize = 12)
axes[1,3].set_xlabel('Time', fontsize = 12)
axes[1,3].set_title('cSWR Pyramidale', fontsize = 15)

figure.set_size_inches([20,10])
figure.savefig(figure_path + 'slope_histogram_rat_'+str(rat_ID_veh[rat_number])+'.png')


####################################################################3

def create_time_histogram(sw_coef_belo):
    bins = np.arange(2,6,0.1)
    histo_output = np.zeros((len(bins)-1,sw_coef_belo.shape[1]))
    for t in range(sw_coef_belo.shape[1]):
        x = np.histogram(sw_coef_belo[:,t],bins = bins)
        histo_output[:,t] = x[0]
    return histo_output

sw_coef_belo_histo = create_time_histogram(sw_intercept_belo)
sw_coef_pyr_histo = create_time_histogram(sw_intercept_pyr)
r_coef_belo_histo = create_time_histogram(r_intercept_belo)
r_coef_pyr_histo = create_time_histogram(r_intercept_pyr)
swr_coef_belo_histo = create_time_histogram(swr_intercept_belo)
swr_coef_pyr_histo = create_time_histogram(swr_intercept_pyr)
cswr_coef_belo_histo = create_time_histogram(cswr_intercept_belo)
cswr_coef_pyr_histo = create_time_histogram(cswr_intercept_pyr)

bins = np.arange(2, 6-0.1 , 0.1)

figure, axes = plt.subplots(2,4)
axes[0,0].pcolormesh(time[select],bins,sw_coef_belo_histo[:,select],cmap = 'Reds')
axes[1,0].pcolormesh(time[select],bins,sw_coef_pyr_histo[:,select],cmap = 'Greys',)
axes[0,0].set_ylabel('Offset', fontsize = 12)
axes[0,0].set_xlabel('Time', fontsize = 12)
axes[0,0].set_title('SW Radiatum', fontsize = 15)
axes[1,0].set_ylabel('Offset', fontsize = 12)
axes[1,0].set_xlabel('Time', fontsize = 12)
axes[1,0].set_title('SW Pyramidale', fontsize = 15)

axes[0, 1].pcolormesh(time[select], bins, r_coef_belo_histo[:, select], cmap='Greys')
axes[1, 1].pcolormesh(time[select], bins, r_coef_pyr_histo[:, select], cmap='Blues' )
axes[0,1].set_ylabel('Offset', fontsize = 12)
axes[0,1].set_xlabel('Time', fontsize = 12)
axes[0,1].set_title('R Radiatum', fontsize = 15)
axes[1,1].set_ylabel('Offset', fontsize = 12)
axes[1,1].set_xlabel('Time', fontsize = 12)
axes[1,1].set_title('R Pyramidale', fontsize = 15)

axes[0, 2].pcolormesh(time[select], bins, swr_coef_belo_histo[:, select], cmap='Purples')
axes[1, 2].pcolormesh(time[select], bins, swr_coef_pyr_histo[:, select], cmap='BuPu')
axes[0,2].set_ylabel('Offset', fontsize = 12)
axes[0,2].set_xlabel('Time', fontsize = 12)
axes[0,2].set_title('SWR Radiatum', fontsize = 15)
axes[1,2].set_ylabel('Offset', fontsize = 12)
axes[1,2].set_xlabel('Time', fontsize = 12)
axes[1,2].set_title('SWR Pyramidale', fontsize = 15)


axes[0, 3].pcolormesh(time[select], bins, cswr_coef_belo_histo[:, select], cmap='OrRd')
axes[1, 3].pcolormesh(time[select], bins, cswr_coef_pyr_histo[:, select], cmap='Oranges')
axes[0,3].set_ylabel('Offset', fontsize = 12)
axes[0,3].set_xlabel('Time', fontsize = 12)
axes[0,3].set_title('cSWR Radiatum', fontsize = 15)
axes[1,3].set_ylabel('Offset', fontsize = 12)
axes[1,3].set_xlabel('Time', fontsize = 12)
axes[1,3].set_title('cSWR Pyramidale', fontsize = 15)

figure.set_size_inches([20,10])
figure.savefig(figure_path + 'offset_histogram_rat_'+str(rat_ID_veh[rat_number])+'.png')


