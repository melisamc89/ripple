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

    fs = 600
    width = 6  # morlet2 width
    low_f = 1  # lowest frequency of interest
    high_f = 200  # highest frequency of interest
    freq = np.linspace(low_f, high_f, num=int(high_f / low_f))  # frequency resolution of time-frequency plot
    freq_log = np.log10(freq)
    widths = width * fs / (2 * freq * np.pi)  # wavelet widths for all frequencies of interest

    vector_intercept_belo = np.zeros((swr_belo.shape[0],3600))
    vector_coef_belo = np.zeros((swr_belo.shape[0],3600))
    freq_index1 = np.where(freq>1)[0]
    freq_index2 = np.where(freq<1000)[0]
    freq_index = np.intersect1d(freq_index1, freq_index2)
    for index in range(swr_belo.shape[0]):
        data_signal = swr_belo[index, :]
        cwtm = signal.cwt(data_signal, signal.morlet2, widths, w=width)

        for t in range(1800 - time_range, 1800 + time_range):
            x = np.log10(np.abs(cwtm[:,t]))
            reg= LinearRegression().fit(freq_log[freq_index].reshape(-1,1), x[freq_index].reshape(-1,1))
            vector_coef_belo[index,t] = reg.coef_[0]
            vector_intercept_belo[index,t] = reg.intercept_[0]

    return vector_intercept_belo,vector_coef_belo



data_path= '/home/melisamc/Documentos/ripple/data/belo_pyr_raw_output/'
data_path_output = '/home/melisamc/Documentos/ripple/data/features_meli/'

figure_path = '/home/melisamc/Documentos/ripple/figures/'

rat_ID_veh = [3,4,9,201,203,206,210,211,213]
rat_ID_cbd= [2,5,10,11,204,205,207,209,212,214]

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



fs = 600
width = 6  # morlet2 width
low_f = 1  # lowest frequency of interest
high_f = 200  # highest frequency of interest
freq = np.linspace(low_f, high_f, num=int(high_f / low_f))  # frequency resolution of time-frequency plot
freq_log = np.log10(freq)
widths = width * fs / (2 * freq * np.pi)  # wavelet widths for all frequencies of interest

freq_index0 = np.where(freq>1)[0]
freq_index1 = np.where(freq>10)[0]
freq_index2 = np.where(freq<1000)[0]
freq_index = np.intersect1d(freq_index1, freq_index2)
freq_index00 = np.intersect1d(freq_index0, freq_index2)

region_list = ['Radiatum','Pyramidale']
type_list = ['sw','r','swr','cswr']
color_list = ['r','b','purple','orange']
color_list2 = ['tomato','royalblue','magenta','brown']
cmaps_list = ['Reds','Blues','Purples','Oranges']
legend1 = ['PRE-Event','Event','Post-Event']
legend2 = ['Spectrum','FitAll','1<Fit','10<Fit']
freq_lims = [1,30]

region = 0
type = 2
data_signal = sw_belo[10, :]
freq_lim = freq_lims[region]
cwtm = signal.cwt(data_signal, signal.morlet2, widths, w=width)


# figure, axes = plt.subplots(2,3)
#
# axes[0,0].plot(np.abs(cwtm[:,1800-300]),c = 'k')
# axes[0,0].plot(np.abs(cwtm[:,1800]),c = color_list [type])
# axes[0,0].plot(np.abs(cwtm[:,1800+300]),'grey')
# axes[0,0].set_xlim([0,200])
# axes[0,0].set_ylim([0,4000])
# axes[0,0].legend(legend1, fontsize = 10)
#
# axes[0,1].plot(np.abs(cwtm[:,1800-300]),c = 'k')
# axes[0,1].plot(np.abs(cwtm[:,1800]),c = color_list[type])
# axes[0,1].plot(np.abs(cwtm[:,1800+300]),c = 'grey')
# axes[0,1].legend(legend1, fontsize = 10)
#
# axes[1,0].plot(np.abs(cwtm[:,1800-300]),c = 'k')
# x = np.log10(np.abs(cwtm[:, 1800-300]))
# reg = LinearRegression().fit(freq_log.reshape(-1, 1), x.reshape(-1, 1))
# y = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
# axes[1,0].plot(freq,y,'k',alpha = 0.50)
# reg = LinearRegression().fit(freq_log[freq_index00].reshape(-1, 1), x[freq_index00].reshape(-1, 1))
# y = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
# axes[1,0].plot(freq,y,'k',alpha = 0.75)
# reg = LinearRegression().fit(freq_log[freq_index].reshape(-1, 1), x[freq_index].reshape(-1, 1))
# y = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
# axes[1,0].plot(freq,y,'k',alpha = 0.9)
# axes[1,0].legend(legend2,fontsize = 10)
#
# axes[1,1].plot(np.abs(cwtm[:,1800+300]),c = 'k')
# x = np.log10(np.abs(cwtm[:, 1800+300]))
# reg = LinearRegression().fit(freq_log.reshape(-1, 1), x.reshape(-1, 1))
# y = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
# axes[1,1].plot(freq,y,'k',alpha = 0.50)
# reg = LinearRegression().fit(freq_log[freq_index00].reshape(-1, 1), x[freq_index00].reshape(-1, 1))
# y = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
# axes[1,1].plot(freq,y,'k',alpha = 0.75)
# reg = LinearRegression().fit(freq_log[freq_index].reshape(-1, 1), x[freq_index].reshape(-1, 1))
# y = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
# axes[1,1].plot(freq,y,'k',alpha = 0.9)
# axes[1,1].legend(legend2,fontsize = 10)
#
# axes[1,2].plot(np.abs(cwtm[:,1800]),c = color_list[type])
# x = np.log10(np.abs(cwtm[:, 1800]))
# reg = LinearRegression().fit(freq_log.reshape(-1, 1), x.reshape(-1, 1))
# y = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
# axes[1,2].plot(freq,y,c = color_list2[type],alpha = 0.5)
# reg = LinearRegression().fit(freq_log[freq_index00].reshape(-1, 1), x[freq_index00].reshape(-1, 1))
# y = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
# axes[1,2].plot(freq,y,c = color_list2[type],alpha = 0.75)
# x = np.log10(np.abs(cwtm[:, 1800]))
# reg = LinearRegression().fit(freq_log[freq_index].reshape(-1, 1), x[freq_index].reshape(-1, 1))
# y = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
# axes[1,2].plot(freq,y,c = color_list2[type],alpha = 0.9)
# axes[1,2].legend(legend2,fontsize = 10)
#
# axes[0,1].set_xscale('log')
# axes[0,1].set_yscale('log')
# for j in range(3):
#     axes[1, j].set_xscale('log')
#     axes[1, j].set_yscale('log')
#     axes[1, j].set_ylim([1,100000])
# axes[0, 1].set_ylim([1,100000])
#
# for i in range(2):
#     for j in range(3):
#         axes[i,j].set_xlabel('Frequency [Hz]')
#         axes[i,j].set_xlabel('Power')
#
# figure.suptitle(type_list[type] +'_' +region_list[region],fontsize =15)
# figure.set_size_inches([12,8])
# figure.savefig(figure_path + 'slope_'+ type_list[type] +'_rat_'+str(rat_ID_veh[rat_number])+'_example_'+region_list[region]+'_regresion.png')

duration_list = []
duration_list.append(duration_sw)
duration_list.append(duration_r)
duration_list.append(duration_swr)
duration_list.append(duration_cswr)

region = 0
type = 0
figure_path = '/home/melisamc/Documentos/ripple/figures/slope/pic_for_gift/' + type_list[type] + '/'
freq_lim = freq_lims[region]
index = 10

time = np.arange(0,cwtm.shape[1])/fs - 3
slope1 = []
slope2 = []
slope3 = []

offset1 = []
offset2 = []
offset3 = []
time_range = []
color_scatter = []


counter = 0
i = -300
index = 0
for i in range(-300,300):
    counter = counter + 1
    t = (1800 + i)/fs - 3
    time_range.append(t)

    figure, axes = plt.subplots(1,5)

    axes[0].pcolormesh(time,freq[freq_lim:],np.abs(cwtm[freq_lim:,:]),cmap = cmaps_list[type])
    axes[0].vlines(t,0,200,color = 'k')
    axes[0].set_xlim([-1,1])
    axes[0].set_ylim([0,200])
    axes[0].set_xlabel('Time(s)',fontsize = 15)
    axes[0].set_ylabel('Frequency(hz)',fontsize = 15)

    if i < -duration_list[type][index][0] or i > duration_list[type][index][1]:
        axes[1].plot(np.abs(cwtm[:,1800+i]),c = 'k')
    else:
        axes[1].plot(np.abs(cwtm[:,1800+i]),c = color_list[type])

    axes[1].set_xlabel('Frequency(Hz)',fontsize = 15)
    axes[1].set_ylabel('Power',fontsize = 15)
    axes[1].set_ylim([0,4000])
    axes[1].set_xlim([0,200])

    x = np.log10(np.abs(cwtm[:, 1800+i]))
    reg = LinearRegression().fit(freq_log.reshape(-1, 1), x.reshape(-1, 1))
    slope1.append(reg.coef_[0])
    offset1.append(reg.intercept_[0])
    y1 = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
    reg = LinearRegression().fit(freq_log[freq_index00].reshape(-1, 1), x[freq_index00].reshape(-1, 1))
    y2 = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
    slope2.append(reg.coef_[0])
    offset2.append(reg.intercept_[0])
    reg = LinearRegression().fit(freq_log[freq_index].reshape(-1, 1), x[freq_index].reshape(-1, 1))
    y3 = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
    slope3.append(reg.coef_[0])
    offset3.append(reg.intercept_[0])
    if i < -duration_list[type][index][0] or i > duration_list[type][index][1]:
        axes[2].plot(np.abs(cwtm[:,1800+i]),c = 'k')
        axes[2].plot(freq,y1,'k',alpha = 0.50)
        axes[2].plot(freq, y2, 'k', alpha=0.75)
        axes[2].plot(freq, y3, 'k', alpha=0.9)
        color_scatter.append('k')
    else:
        axes[2].plot(np.abs(cwtm[:,1800+i]),c = color_list[type])
        axes[2].plot(freq,y1,c = color_list2[type],alpha = 0.50)
        axes[2].plot(freq, y2,c = color_list2[type], alpha=0.75)
        axes[2].plot(freq, y3,c = color_list2[type], alpha=0.9)
        color_scatter.append(color_list2[type])

    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_ylim([1,100000])
    axes[2].set_xlim([1,1000])
    axes[2].set_xlabel('Frequency(Hz)',fontsize = 15)
    axes[2].set_ylabel('Power',fontsize = 15)


    axes[3].scatter(time_range, slope1, c = color_scatter, alpha = 0.3,s = 1)
    axes[3].scatter(time_range, slope2, c = color_scatter, alpha = 0.5,s = 1)
    axes[3].scatter(time_range, slope3, c = color_scatter, alpha = 0.7,s = 1)
    axes[3].set_ylim([-3,0])
    axes[3].set_xlim([-1,1])
    axes[3].set_xlabel('Time (s)',fontsize = 15)
    axes[3].set_ylabel('Slope',fontsize = 15)
    axes[3].legend(['AllFit','1<Fit','10<Fit'],fontsize = 15)


    axes[4].scatter(time_range, offset1, c = color_scatter, alpha = 0.3,s = 1)
    axes[4].scatter(time_range, offset2, c = color_scatter, alpha = 0.5,s = 1)
    axes[4].scatter(time_range, offset3, c = color_scatter, alpha = 0.7,s = 1)
    axes[4].set_ylim([1,8])
    axes[4].set_xlim([-1,1])
    axes[4].set_xlabel('Time (s)',fontsize = 15)
    axes[4].set_ylabel('Offset',fontsize = 15)
    axes[4].legend(['AllFit','1<Fit','10<Fit'],fontsize = 15)

    figure.suptitle(type_list[type] +'_' +region_list[region],fontsize =20)
    figure.set_size_inches([26,5])
    figure.savefig(figure_path + 'slope_'+ type_list[type] +'_rat_'+str(rat_ID_veh[rat_number])+'_example_'+region_list[region]+'_regresion_10000000'+str(counter+100)+'.png',bbox_inches='tight')
    plt.close()


##############################3
###MEAN PLOT###################
##############################

region = 1
type = 2
freq_lim = freq_lims[region]
figure_path = '/home/melisamc/Documentos/ripple/figures/slope/pic_for_gift/' + type_list[type] + '/'
index = 10

cwtm_ = np.zeros((len(freq),3601))
for index in range(swr_pyr.shape[0]):
    data_signal = swr_pyr[index, :]
    cwtm = signal.cwt(data_signal, signal.morlet2, widths, w=width)
    cwtm_ = np.add(cwtm_,cwtm)

cwtm_ = np.divide(cwtm_,index)
cwtm = cwtm_


time = np.arange(0,cwtm.shape[1])/fs - 3
slope1 = []
slope2 = []
slope3 = []

offset1 = []
offset2 = []
offset3 = []
time_range = []
color_scatter = []
duration_pre = np.mean(duration_list[type][:][0])
duration_post = np.mean(duration_list[type][:][1])

counter = 0
i = -300
index = 0
for i in range(-300,300):
    counter = counter + 1
    t = (1800 + i)/fs - 3
    time_range.append(t)

    figure, axes = plt.subplots(2,2)

    axes[0,0].pcolormesh(time,freq[freq_lim:],np.abs(cwtm[freq_lim:,:]),cmap = cmaps_list[type])
    axes[0,0].vlines(t,0,200,color = 'k')
    axes[0,0].set_xlim([-1,1])
    axes[0,0].set_ylim([0,200])
    axes[0,0].set_xlabel('Time(s)',fontsize = 15)
    axes[0,0].set_ylabel('Frequency(hz)',fontsize = 15)


    x = np.log10(np.abs(cwtm[:, 1800+i]))
    reg = LinearRegression().fit(freq_log.reshape(-1, 1), x.reshape(-1, 1))
    slope1.append(reg.coef_[0])
    offset1.append(reg.intercept_[0])
    y1 = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
    reg = LinearRegression().fit(freq_log[freq_index00].reshape(-1, 1), x[freq_index00].reshape(-1, 1))
    y2 = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
    slope2.append(reg.coef_[0])
    offset2.append(reg.intercept_[0])
    reg = LinearRegression().fit(freq_log[freq_index].reshape(-1, 1), x[freq_index].reshape(-1, 1))
    y3 = 10**(reg.coef_[0] * freq_log + reg.intercept_[0])
    slope3.append(reg.coef_[0])
    offset3.append(reg.intercept_[0])
    if i < -duration_pre or i > duration_post:
        axes[0,1].plot(np.abs(cwtm[:,1800+i]),c = 'k')
        axes[0,1].plot(freq,y1,'k',alpha = 0.50)
        axes[0,1].plot(freq, y2, 'k', alpha=0.75)
        axes[0,1].plot(freq, y3, 'k', alpha=0.9)
        color_scatter.append('k')
    else:
        axes[0,1].plot(np.abs(cwtm[:,1800+i]),c = color_list[type])
        axes[0,1].plot(freq,y1,c = color_list2[type],alpha = 0.50)
        axes[0,1].plot(freq, y2,c = color_list2[type], alpha=0.75)
        axes[0,1].plot(freq, y3,c = color_list2[type], alpha=0.9)
        color_scatter.append(color_list2[type])

    axes[0,1].set_xscale('log')
    axes[0,1].set_yscale('log')
    axes[0,1].set_ylim([1,10000])
    axes[0,1].set_xlim([1,1000])
    axes[0,1].set_xlabel('Frequency(Hz)',fontsize = 15)
    axes[0,1].set_ylabel('Power',fontsize = 15)


    axes[1,0].scatter(time_range, slope1, c = color_scatter, alpha = 0.3,s = 1)
    axes[1,0].scatter(time_range, slope2, c = color_scatter, alpha = 0.5,s = 1)
    axes[1,0].scatter(time_range, slope3, c = color_scatter, alpha = 0.7,s = 1)
    axes[1,0].set_ylim([-3,0])
    axes[1,0].set_xlim([-1,1])
    axes[1,0].set_xlabel('Time (s)',fontsize = 15)
    axes[1,0].set_ylabel('Slope',fontsize = 15)
    axes[1,0].legend(['AllFit','1<Fit','10<Fit'],fontsize = 15)


    axes[1,1].scatter(time_range, offset1, c = color_scatter, alpha = 0.3,s = 1)
    axes[1,1].scatter(time_range, offset2, c = color_scatter, alpha = 0.5,s = 1)
    axes[1,1].scatter(time_range, offset3, c = color_scatter, alpha = 0.7,s = 1)
    axes[1,1].set_ylim([8,8])
    axes[1,1].set_xlim([-1,1])
    axes[1,1].set_xlabel('Time (s)',fontsize = 15)
    axes[1,1].set_ylabel('Offset',fontsize = 15)
    axes[1,1].legend(['AllFit','1<Fit','10<Fit'],fontsize = 15)

    figure.suptitle(type_list[type] +'_' +region_list[region],fontsize =20)
    figure.set_size_inches([7,7])
    figure.savefig(figure_path + 'slope_'+ type_list[type] +'_rat_'+str(rat_ID_veh[rat_number])+'_example_'+region_list[region]+'_regresion_10000000'+str(counter+100)+'.png',bbox_inches='tight')
    plt.close()
#
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

fs = 600
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

########################
figure, axes = plt.subplots(1,2)
axes[0].plot(time[select],np.mean(sw_coef_belo[:,select],axis=0),c = 'r',alpha = 0.8)
axes[1].plot(time[select],np.mean(sw_coef_pyr[:,select],axis=0),c = 'k',alpha = 0.8)
axes[0].plot(time[select],np.mean(r_coef_belo[:,select],axis=0),c = 'k',alpha = 0.8)
axes[1].plot(time[select],np.mean(r_coef_pyr[:,select],axis=0),c = 'b',alpha = 0.8)
axes[0].plot(time[select],np.mean(swr_coef_belo[:,select],axis=0),c = 'darkviolet',alpha = 0.8)
axes[1].plot(time[select],np.mean(swr_coef_pyr[:,select],axis=0),c = 'violet',alpha = 0.8)
axes[0].plot(time[select],np.mean(cswr_coef_belo[:,select],axis = 0),c = 'darkorange',alpha = 0.8)
axes[1].plot(time[select],np.mean(cswr_coef_pyr[:,select],axis=0),c = 'orange',alpha = 0.8)

axes[0].set_ylabel('Slope', fontsize = 12)
axes[0].set_xlabel('Time', fontsize = 12)
axes[0].set_title('Radiatum', fontsize = 15)
axes[0].set_ylim([-2.5,0])
axes[0].legend(['SW','R','SWR','cSWR'],fontsize = 12)

axes[1].set_ylabel('Slope', fontsize = 12)
axes[1].set_xlabel('Time', fontsize = 12)
axes[1].set_title('Pyramidale', fontsize = 15)
axes[1].set_ylim([-2.5,0])
axes[1].legend(['SW','R','SWR','cSWR'],fontsize = 12)

figure.set_size_inches([10,5])
figure.savefig(figure_path + 'slope_mean_rat_'+str(rat_ID_veh[rat_number])+'2.png')
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
##################################################
figure, axes = plt.subplots(1,2)
axes[0].plot(time[select],np.mean(sw_intercept_belo[:,select],axis=0),c = 'r',alpha = 0.8)
axes[1].plot(time[select],np.mean(sw_intercept_pyr[:,select],axis=0),c = 'k',alpha = 0.8)
axes[0].plot(time[select],np.mean(r_intercept_belo[:,select],axis=0),c = 'k',alpha = 0.8)
axes[1].plot(time[select],np.mean(r_intercept_pyr[:,select],axis=0),c = 'b',alpha = 0.8)
axes[0].plot(time[select],np.mean(swr_intercept_belo[:,select],axis=0),c = 'darkviolet',alpha = 0.8)
axes[1].plot(time[select],np.mean(swr_intercept_pyr[:,select],axis=0),c = 'violet',alpha = 0.8)
axes[0].plot(time[select],np.mean(cswr_intercept_belo[:,select],axis = 0),c = 'darkorange',alpha = 0.8)
axes[1].plot(time[select],np.mean(cswr_intercept_pyr[:,select],axis=0),c = 'orange',alpha = 0.8)

axes[0].set_ylabel('Offset', fontsize = 12)
axes[0].set_xlabel('Time', fontsize = 12)
axes[0].set_title('Radiatum', fontsize = 15)
axes[0].set_ylim([2,6])
axes[0].legend(['SW','R','SWR','cSWR'],fontsize = 12)

axes[1].set_ylabel('Offset', fontsize = 12)
axes[1].set_xlabel('Time', fontsize = 12)
axes[1].set_title('Pyramidale', fontsize = 15)
axes[1].set_ylim([2,6])
axes[1].legend(['SW','R','SWR','cSWR'],fontsize = 12)

figure.set_size_inches([10,5])
figure.savefig(figure_path + 'offset_mean_rat_'+str(rat_ID_veh[rat_number])+'2.png')
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


