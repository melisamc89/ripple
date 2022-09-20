
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle


rat_ID_veh = [3,4,9,201,203,206,210,211,213]
rat_ID_cbd= [2,5,10,11,204,205,207,209,212,214]

keywords_veh = ['HPCpyra_complex_swr_veh','HPCpyra_ripple_veh','HPCpyra_swr_veh']
keywords_cbd = ['HPCpyra_complex_swr_cbd','HPCpyra_ripple_cbd','HPCpyra_swr_cbd']

type_label = ['ComplexRipple','Ripple','SWR']
srate = 600
rat_number = 0

data_path_veh = '/home/melisamc/Documentos/ripple/data/HPCpyra/'
data_path_cbd = '/home/melisamc/Documentos/ripple/data/CBD/HPCpyra/'
figure_path = '/home/melisamc/Documentos/ripple/figures/'

color_list = ['b','r','g']
time = np.arange(1500, 2000)
time2 = np.arange(225, 375)


mean_matrix_veh = np.zeros((len(rat_ID_veh),3, time.shape[0]))
mean_matrix_cbd = np.zeros((len(rat_ID_cbd),3, time.shape[0]))
mean_matrix_cbd_early = np.zeros((len(rat_ID_cbd),3, time.shape[0]))
mean_matrix_cbd_late = np.zeros((len(rat_ID_cbd),3, time.shape[0]))

for rat_number in range(len(rat_ID_veh)):
    data_file = 'HPCpyra_events_ratID' + str(rat_ID_veh[rat_number]) + '.mat'
    data = sio.loadmat(data_path_veh + data_file)
    for type1 in range(2):
        for type2 in range(2):
            index = type1*2+type2
            if index < 3:
                ripple_type = data[keywords_veh[index]]
                mean_ripple = np.mean(ripple_type[:,time],axis=0)
                mean_matrix_veh[rat_number,index,:] = mean_ripple

for rat_number in range(len(rat_ID_cbd)):
    data_file = 'HPCpyra_events_ratID' + str(rat_ID_cbd[rat_number]) + '.mat'
    data = sio.loadmat(data_path_cbd + data_file)
    for type1 in range(2):
        for type2 in range(2):
            index = type1*2+type2
            if index < 3:
                ripple_type = data[keywords_cbd[index]]
                mean_ripple = np.nanmean(ripple_type[:,time],axis=0)
                mean_matrix_cbd[rat_number,index,:] = mean_ripple

                ripple_type_early = ripple_type[0:int(ripple_type.shape[0]/2),:]
                ripple_type_late = ripple_type[int(ripple_type.shape[0]/2):,:]
                mean_ripple_early = np.nanmean(ripple_type_early[:,time],axis=0)
                mean_ripple_late = np.nanmean(ripple_type_late[:,time],axis=0)
                mean_matrix_cbd_early[rat_number,index,:] = mean_ripple_early
                mean_matrix_cbd_late[rat_number,index,:] = mean_ripple_late

mean_ripple_matrix_veh = np.nanmean(mean_matrix_veh,axis = 0)
std_ripple_matrix_veh = np.nanstd(mean_matrix_veh,axis = 0)
mean_ripple_matrix_cbd = np.nanmean(mean_matrix_cbd,axis = 0)
std_ripple_matrix_cbd = np.nanstd(mean_matrix_cbd,axis = 0)

mean_ripple_matrix_cbd_early = np.nanmean(mean_matrix_cbd_early,axis = 0)
std_ripple_matrix_cbd_early = np.nanstd(mean_matrix_cbd_early,axis = 0)

mean_ripple_matrix_cbd_late = np.nanmean(mean_matrix_cbd_late,axis = 0)
std_ripple_matrix_cbd_late = np.nanstd(mean_matrix_cbd_late,axis = 0)

figure, axes = plt.subplots(3,2)

for i in range(3):
    axes[i,0].plot(time[time2]/srate-3,mean_ripple_matrix_veh[i,time2],color='k')
    axes[i,0].plot(time[time2] / srate - 3, mean_ripple_matrix_cbd[i, time2], color=color_list[i])

    axes[i,0].fill_between(time[time2]/srate-3,mean_ripple_matrix_veh[i,time2]-std_ripple_matrix_veh[i,time2],mean_ripple_matrix_veh[i,time2]+std_ripple_matrix_veh[i,time2],color='k',alpha=0.3)
    axes[i,0].fill_between(time[time2]/srate-3,mean_ripple_matrix_cbd[i,time2]-std_ripple_matrix_cbd[i,time2],mean_ripple_matrix_cbd[i,time2]+std_ripple_matrix_cbd[i,time2],color=color_list[i],alpha=0.3)

    axes[i,0].set_ylabel('Act [units?]', fontsize = 12)
    axes[i,0].set_title(type_label[i], fontsize=12)
    axes[i,0].legend(['Vehicle','CBD'],fontsize = 8)

    axes[i,1].plot(time[time2]/srate-3,mean_ripple_matrix_cbd_early[i,time2],color='k')
    axes[i,1].plot(time[time2] / srate - 3, mean_ripple_matrix_cbd_late[i, time2], color=color_list[i])

    axes[i,1].fill_between(time[time2]/srate-3,mean_ripple_matrix_cbd_early[i,time2]-std_ripple_matrix_cbd_early[i,time2],mean_ripple_matrix_cbd_early[i,time2]+std_ripple_matrix_cbd_early[i,time2],color='k',alpha=0.3)
    axes[i,1].fill_between(time[time2]/srate-3,mean_ripple_matrix_cbd_late[i,time2]-std_ripple_matrix_cbd_late[i,time2],mean_ripple_matrix_cbd_late[i,time2]+std_ripple_matrix_cbd_late[i,time2],color=color_list[i],alpha=0.3)

    axes[i,1].set_ylabel('Act [units?]', fontsize = 12)
    axes[i,1].set_title(type_label[i], fontsize=12)
    axes[i,1].legend(['Early','Late'],fontsize = 8)

axes[2,0].set_xlabel('Time[s]', fontsize = 15)
axes[2,1].set_xlabel('Time[s]', fontsize = 15)
figure.set_size_inches([25,15])
figure.savefig(figure_path + 'vehicle_vs_cbd_early_and_late.png')
