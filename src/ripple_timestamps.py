
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from scipy import signal
from scipy.io import savemat

def count_events(data_path,ID_sequence,events_sequence):
    events_counter = np.zeros((4, len(ID_sequence)))
    for rat_number in range(len(ID_sequence)):
        data_file = 'HPCpyra_events_ratID' + str(ID_sequence[rat_number]) + '.mat'
        data = sio.loadmat(data_path + data_file)
        for key in range(3):
            ripple_data = data[events_sequence[key]]
            events_counter[key, rat_number] += ripple_data.shape[0]
            events_counter[3, rat_number] += ripple_data.shape[0]
    total_ripple_count_veh = np.sum(events_counter, axis=1)
    return total_ripple_count_veh, events_counter

def load_all_data(data_path,ID_sequence,events_sequence,total_count,time2):

    ripple_veh_matrix_list = []
    for i in range(4):
        ripple_veh_matrix_list.append(np.zeros((int(total_count[i]),len(time2))))
    class_vector = np.zeros((int(total_count[3]),))

    ###Vehicle
    counter = np.zeros((4,))
    for rat_number in range(len(ID_sequence)):
        data_file = 'HPCpyra_events_ratID' + str(ID_sequence[rat_number]) + '.mat'
        data = sio.loadmat(data_path + data_file)
        for key in range(3):
            ripple_data = data[events_sequence[key]]
            ripple_veh_matrix_list[key][int(counter[key]):int(counter[key])+ripple_data.shape[0]]= ripple_data[:,time[time2]]
            class_vector[int(counter[3]):int(counter[3]+ripple_data.shape[0])] = key
            ripple_veh_matrix_list[3][int(counter[3]):int(counter[3] + ripple_data.shape[0]), :] = ripple_data[:,time[time2]]
            counter[key]+=ripple_data.shape[0]
            counter[3]+=ripple_data.shape[0]

    return ripple_veh_matrix_list, class_vector

def load_bin_information(data_path,ID_sequence,events_sequence,total_count,desired_variable):

    ###Vehicle
    bin_vector = np.zeros((total_count,))
    counter = 0
    for rat_number in range(len(ID_sequence)):
        data_file = 'GC_ratID' + str(ID_sequence[rat_number]) + '_veh.mat'
        data = sio.loadmat(data_path + data_file)
        data = data['data']
        for key in range(3):
            ripple_info_bin = data[events_sequence[key]][0][0][:, desired_variable]
            bin_vector[counter:counter+ripple_info_bin.shape[0]]=ripple_info_bin
            counter+=ripple_info_bin.shape[0]

    return bin_vector

def extract_bin_information(data_path,output_path,ID_sequence,events_sequence,desired_variable):

    for rat_number in range(len(ID_sequence)):
        input_data_file = 'GC_ratID' + str(ID_sequence[rat_number]) + '_cbd.mat'
        output_file_name = output_path + 'GC_ratID' + str(ID_sequence[rat_number]) + '_cbd.mat'

        input_data = sio.loadmat(data_path + input_data_file)
        data = input_data['data']
        information_list = []

        for key in range(3):
            ripple_info_bin = data[events_sequence[key]][0][0][:, desired_variable]
            information_list.append(ripple_info_bin)
        output = {events_sequence[0]:information_list[0],events_sequence[1]:information_list[1],events_sequence[2]:information_list[2]}
        savemat(output_file_name,output)
    return

rat_ID_veh = [3,4,9,201,203,206,210,211,213]
rat_ID_cbd= [2,5,10,11,204,205,207,209,212,214]

keywords= ['cr','r','swr']
keywords_veh = ['HPCpyra_complex_swr_veh','HPCpyra_ripple_veh','HPCpyra_swr_veh']
keywords_cbd = ['HPCpyra_complex_swr_cbd','HPCpyra_ripple_cbd','HPCpyra_swr_cbd']
type_label = ['ComplexRipple','Ripple','SWR']
srate = 600

data_path_veh = '/home/melisamc/Documentos/ripple/data/HPCpyra/'
data_path_cbd = '/home/melisamc/Documentos/ripple/data/CBD/HPCpyra/'
figure_path = '/home/melisamc/Documentos/ripple/figures/'

# data_path = '/home/melisamc/Documentos/ripple/data/group_info/'
# output_path_veh = '/home/melisamc/Documentos/ripple/data/bins_information/VEH/'
# extract_bin_information(data_path,output_path_veh,rat_ID_cbd,keywords,desired_variable = 6)

time = np.arange(0, 3600)
time2 = np.arange(0, 3600)
time = np.arange(1500, 2000)
time2 = np.arange(225, 375)

total_ripple_count_veh, total_matrix_veh= count_events(data_path_veh,rat_ID_veh,keywords_veh)
percetage_ripple_count_veh = total_ripple_count_veh / total_ripple_count_veh[3]
# total_ripple_count_cbd, total_matrix_cbd = count_events(data_path_cbd,rat_ID_cbd,keywords_cbd)
# percetage_ripple_count_cbd = total_ripple_count_cbd / total_ripple_count_cbd[3]
ripple_veh_matrix_list, class_vector_veh= load_all_data(data_path_veh,rat_ID_veh,keywords_veh,total_ripple_count_veh,time2)
# ripple_cbd_matrix_list, class_vector_cbd = load_all_data(data_path_cbd,rat_ID_cbd,keywords_cbd,total_ripple_count_cbd,time2)


data_path= '/home/melisamc/Documentos/ripple/data/group_info/'

bin_vector = load_bin_information(data_path, rat_ID_veh, keywords, int(total_ripple_count_veh[3]),desired_variable = 6)
time_vector = load_bin_information(data_path, rat_ID_veh, keywords, int(total_ripple_count_veh[3]),desired_variable = 1)

figure = plt.figure()
gs = plt.GridSpec(8, 3)
time_srate = 1200
for i in range(3):
    axes0 = figure.add_subplot(gs[0, i])#, projection='3d')
    time_data = time_vector[np.where(class_vector_veh==i)[0]] / time_srate
    axes0.hist(time_data,bins = 20)#density = True)
    axes0.set_ylim([0,400])
    axes0.set_title(type_label[i],fontsize = 20)
    axes0.set_xlabel('Time bin [s]',fontsize = 15)
    axes0.set_ylabel('Count',fontsize = 15)

    axes00 = figure.add_subplot(gs[2, i])#, projection='3d')
    irt = np.diff(time_data)
    irt = irt[np.where(irt>0)[0]]
    axes00.hist(irt,bins = 50)#density = True)
    axes00.set_xlabel('InterRippleTime [s]',fontsize = 15)
    axes00.set_ylabel('Count',fontsize = 15)
    axes00.set_yscale('log')
    axes00.set_ylim([0,1000])
    axes00.set_xlim([0,1000])

    #axes00.set_xscale('log')

    axes1 = figure.add_subplot(gs[4:8, i])
    axes1.set_xlabel('Time [s]',fontsize = 15)
    axes1.set_ylabel('TEMPORAL BIN',fontsize = 15)
    axes1.set_title('Total = '+ str(len(time_data)),fontsize = 12)

    for bin in range(11):
        index = np.logical_and(class_vector_veh==i,bin_vector ==bin)
        ripple_bin = ripple_veh_matrix_list[3][index,:]
        mean_ripple_mean = np.nanmean(ripple_bin,axis = 0)
        if np.sum(mean_ripple_mean):
            norm_version = (mean_ripple_mean - np.min(mean_ripple_mean))/np.max(mean_ripple_mean)
            axes1.plot(time2/srate,norm_version+bin)
figure.set_size_inches([15,10])
figure.savefig(figure_path + 'temporal_binning_time_cbd.png')
plt.show()


irt = np.diff(time_vector,prepend=0) / time_srate
bins_nums = np.arange(0,600,20)
figure, axes = plt.subplots(2,6)
for n1 in range(2):
    for n2 in range(6):
        i = n1*6+n2 +1
        irt_data = irt[np.where(bin_vector== i)[0]]
        irt_data = irt_data[np.where(irt_data>0)[0]]
        subset = np.random.random(460)*len(irt_data)
        irt_data

        if np.sum(irt_data):
            axes[n1,n2].hist(irt_data,bins=bins_nums,density = True)
            #axes[n1,n2].set_yscale('log')
            axes[n1,n2].set_xlim([-10,500])
            axes[n1,n2].set_ylim([0.00001,0.01])

        axes[n1,n2].set_title('BIN = '+str(i)+ ' Count='+str(len(irt_data)),fontsize = 12)
        axes[n1,n2].set_xlabel('Time(s)',fontsize = 12)
        axes[n1,n2].set_ylabel('Density',fontsize = 12)

figure.set_size_inches([20,10])
figure.savefig(figure_path + 'irt_in_bins_random_cbd.png')
plt.show()

figure, axes = plt.subplots(2,6)
bins_nums = np.arange(0, 600, 50)
for i in range(3):
    class_data = np.where(class_vector_veh==i)[0]
    time_data = time_vector[class_data] / time_srate
    bin_class = bin_vector[class_data]
    irt = np.diff(time_data, prepend=0)
    for n1 in range(2):
        for n2 in range(6):
            bin = n1*6 + n2 +1
            index = np.where(bin_class==bin)[0]
            irt_data = irt[index]
            irt_data = irt_data[np.where(irt_data > 0)[0]]
            if np.sum(irt_data):
                axes[n1,n2].hist(irt_data, bins=bins_nums, alpha = 0.5,density = True)
                #axes[n1,n2].set_yscale('log')
                axes[n1,n2].set_xlim([-10, 500])
                axes[n1,n2].set_ylim([0.00000001, 0.02])

            axes[n1,n2].set_title('BIN = '+str(bin),fontsize = 12)
            axes[n1,n2].set_xlabel('Time(s)',fontsize = 12)
            axes[n1, n2].legend(type_label)

            #axes[n1,n2].set_ylabel('Density',fontsize = 12)

figure.set_size_inches([20,10])
figure.savefig(figure_path + 'irt_in_bins_class_cbd.png')

plt.show()
