
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

def load_bin_information(data_path,ID_sequence,events_sequence,total_count,group,desired_variable):

    ###Vehicle
    bin_vector = np.zeros((total_count,))
    rat_id_output = np.zeros((total_count,))
    counter = 0
    for rat_number in range(len(ID_sequence)):
        data_file = 'GC_ratID' + str(ID_sequence[rat_number]) + '_'+group+'.mat'
        data = sio.loadmat(data_path + data_file)
        data = data['data']
        for key in range(3):
            ripple_info_bin = data[events_sequence[key]][0][0][:, desired_variable]
            bin_vector[counter:counter+ripple_info_bin.shape[0]]=ripple_info_bin
            rat_id_output[counter:counter+ripple_info_bin.shape[0]] = np.ones((ripple_info_bin.shape[0],))*rat_number
            counter+=ripple_info_bin.shape[0]

    return bin_vector, rat_id_output

def extract_bin_information(data_path,output_path,ID_sequence,events_sequence,group,desired_variable):

    for rat_number in range(len(ID_sequence)):
        input_data_file = 'GC_ratID' + str(ID_sequence[rat_number]) + group + '.mat'
        output_file_name = output_path + 'GC_ratID' + str(ID_sequence[rat_number]) + '_'+group+'.mat'

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
total_ripple_count_cbd, total_matrix_cbd = count_events(data_path_cbd,rat_ID_cbd,keywords_cbd)
percetage_ripple_count_cbd = total_ripple_count_cbd / total_ripple_count_cbd[3]
ripple_veh_matrix_list, class_vector_veh= load_all_data(data_path_veh,rat_ID_veh,keywords_veh,total_ripple_count_veh,time2)
ripple_cbd_matrix_list, class_vector_cbd = load_all_data(data_path_cbd,rat_ID_cbd,keywords_cbd,total_ripple_count_cbd,time2)

data_path= '/home/melisamc/Documentos/ripple/data/group_info/'
bin_vector_veh, rat_vector_veh = load_bin_information(data_path, rat_ID_veh, keywords, int(total_ripple_count_veh[3]),'veh',desired_variable = 6)
time_vector_veh, rat_vector_veh = load_bin_information(data_path, rat_ID_veh, keywords, int(total_ripple_count_veh[3]),'veh',desired_variable = 1)
bin_vector_cbd, rat_vector_cbd = load_bin_information(data_path, rat_ID_cbd, keywords, int(total_ripple_count_cbd[3]),'cbd',desired_variable = 6)
time_vector_cbd, rat_vector_cbd = load_bin_information(data_path, rat_ID_cbd, keywords, int(total_ripple_count_cbd[3]),'cbd',desired_variable = 1)

figure = plt.figure()
gs = plt.GridSpec(8, 3)
time_srate = 1200
for i in range(3):
    axes0 = figure.add_subplot(gs[0, i])#, projection='3d')
    time_data = time_vector_veh[np.where(class_vector_veh==i)[0]] / time_srate
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
        index = np.logical_and(class_vector_veh==i,bin_vector_veh ==bin)
        ripple_bin = ripple_veh_matrix_list[3][index,:]
        mean_ripple_mean = np.nanmean(ripple_bin,axis = 0)
        if np.sum(mean_ripple_mean):
            norm_version = (mean_ripple_mean - np.min(mean_ripple_mean))/np.max(mean_ripple_mean)
            axes1.plot(time2/srate,norm_version+bin)
figure.set_size_inches([15,10])
figure.savefig(figure_path + 'temporal_binning_time_veh.png')
plt.show()


irt = np.diff(time_vector_veh,prepend=0) / time_srate
bins_nums = np.arange(0,600,20)
figure, axes = plt.subplots(2,6)
for n1 in range(2):
    for n2 in range(6):
        i = n1*6+n2 +1
        irt_data = irt[np.where(bin_vector_veh== i)[0]]
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
figure.savefig(figure_path + 'irt_in_bins_random_veh.png')
plt.show()

figure, axes = plt.subplots(2,6)
bins_nums = np.arange(0, 600, 50)
for i in range(3):
    class_data = np.where(class_vector_veh==i)[0]
    time_data = time_vector_veh[class_data] / time_srate
    bin_class = bin_vector_veh[class_data]
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
figure.savefig(figure_path + 'irt_in_bins_class_veh.png')

plt.show()

#############################3

#ripple_transitions

transition_matrix_veh = np.zeros((3,3))
transition_matrix_veh_random = np.zeros((3,3))

for rat in range(len(rat_ID_veh)):
    index = np.where(rat_vector_veh == rat)[0]
    time_data = time_vector_veh[index]
    sorted_index = np.argsort(time_data)
    time_data = time_data[sorted_index]
    classification_data = class_vector_veh[index]
    classification_data = classification_data[sorted_index]
    permutation = np.random.permutation(len(classification_data))
    classification_data_perm = classification_data[permutation]
    for number in range(len(classification_data)-1):
        val_i = int(classification_data[number])
        val_j = int(classification_data[number+1])
        transition_matrix_veh[val_i,val_j]+=1
        val_i = int(classification_data_perm[number])
        val_j = int(classification_data_perm[number+1])
        transition_matrix_veh_random[val_i,val_j]+=1

for i in range(3):
    transition_matrix_veh[i,:] = transition_matrix_veh[i,:]/np.sum(transition_matrix_veh[i,:])
    transition_matrix_veh_random[i,:] = transition_matrix_veh_random[i,:]/np.sum(transition_matrix_veh_random[i,:])

transition_matrix_cbd = np.zeros((3,3))
transition_matrix_cbd_random = np.zeros((3,3))

for rat in range(len(rat_ID_cbd)):
    index = np.where(rat_vector_cbd == rat)[0]
    time_data = time_vector_cbd[index]
    sorted_index = np.argsort(time_data)
    time_data = time_data[sorted_index]
    classification_data = class_vector_cbd[index]
    classification_data = classification_data[sorted_index]
    permutation = np.random.permutation(len(classification_data))
    classification_data_perm = classification_data[permutation]

    for number in range(len(classification_data)-1):
        val_i = int(classification_data[number])
        val_j = int(classification_data[number+1])
        transition_matrix_cbd[val_i,val_j]+=1
        val_i = int(classification_data_perm[number])
        val_j = int(classification_data_perm[number+1])
        transition_matrix_cbd_random[val_i,val_j]+=1

for i in range(3):
    transition_matrix_cbd[i,:] = transition_matrix_cbd[i,:]/np.sum(transition_matrix_cbd[i,:])
    transition_matrix_cbd_random[i,:] = transition_matrix_cbd_random[i,:]/np.sum(transition_matrix_cbd_random[i,:])


dx, dy = 1, 1
y, x = np.mgrid[slice(0, 2 + dy, dy),
                slice(0, 2 + dx, dx)]

ripple_label = ['CSWR','R','SWR']
figure, axes = plt.subplots(2,2)
orig_map=plt.cm.get_cmap('gray')
reversed_map = orig_map.reversed()

ax0 = axes[0,0]
pcm0 = ax0.pcolormesh(x,y,transition_matrix_veh,cmap=reversed_map,vmin = 0, vmax = 0.7)
figure.colorbar(pcm0, ax=ax0)
ax0.set_xticks(np.arange(0,3))
ax0.set_xticklabels(ax0.get_xticks(), rotation = 45)
ax0.set_xticklabels(ripple_label)
ax0.set_yticks(np.arange(0,3))
ax0.set_yticklabels(ripple_label)
ax0.set_title('Vehicle',fontsize = 15)

ax0 = axes[1,0]
pcm0 = ax0.pcolormesh(x,y,transition_matrix_veh_random,cmap=reversed_map,vmin = 0, vmax = 0.7)
figure.colorbar(pcm0, ax=ax0)
ax0.set_xticks(np.arange(0,3))
ax0.set_xticklabels(ax0.get_xticks(), rotation = 45)
ax0.set_xticklabels(ripple_label)
ax0.set_yticks(np.arange(0,3))
ax0.set_yticklabels(ripple_label)
ax0.set_title('VehicleRANDOM',fontsize = 15)

ax1 = axes[0,1]
orig_map=plt.cm.get_cmap('gray')
reversed_map = orig_map.reversed()
pcm0 = ax1.pcolormesh(x,y,transition_matrix_cbd,cmap=reversed_map,vmin = 0, vmax = 0.7)
figure.colorbar(pcm0, ax=ax1)
ax1.set_xticks(np.arange(0,3))
ax1.set_xticklabels(ax1.get_xticks(), rotation = 45)
ax1.set_xticklabels(ripple_label)
ax1.set_yticks(np.arange(0,3))
ax1.set_yticklabels(ripple_label)
ax1.set_title('CBD',fontsize = 15)

ax1 = axes[1,1]
orig_map=plt.cm.get_cmap('gray')
reversed_map = orig_map.reversed()
pcm0 = ax1.pcolormesh(x,y,transition_matrix_cbd_random,cmap=reversed_map,vmin = 0, vmax = 0.7)
figure.colorbar(pcm0, ax=ax1)
ax1.set_xticks(np.arange(0,3))
ax1.set_xticklabels(ax1.get_xticks(), rotation = 45)
ax1.set_xticklabels(ripple_label)
ax1.set_yticks(np.arange(0,3))
ax1.set_yticklabels(ripple_label)
ax1.set_title('CBD RANDOM',fontsize = 15)

figure.suptitle('Ripple transition Matrix',fontsize = 20)
figure.set_size_inches([10,10])
figure.savefig(figure_path + 'ripple_transition_matrix_test.png')

plt.show()

#################################################
#transition matrix by bins
N_perm = 100
transition_matrix_veh = np.zeros((12,3,3))
transition_matrix_veh_random_ = np.zeros((N_perm,12,3,3))

for rat in range(len(rat_ID_veh)):
    index = np.where(rat_vector_veh == rat)[0]
    time_data = time_vector_veh[index]
    sorted_index = np.argsort(time_data)
    time_data = time_data[sorted_index]
    classification_data = class_vector_veh[index]
    classification_data = classification_data[sorted_index]
    bin_data = bin_vector_veh[index]
    bin_data = bin_data[sorted_index]
    for bin in range(0,12):
        bin_index = np.where(bin_data==bin)[0]
        classification_data_bin = classification_data[bin_index]
        for number in range(len(classification_data_bin)-1):
            val_i = int(classification_data_bin[number])
            val_j = int(classification_data_bin[number+1])
            transition_matrix_veh[bin,val_i,val_j]+=1
        for n in range(N_perm):
            permutation = np.random.permutation(len(classification_data_bin))
            classification_data_perm = classification_data_bin[permutation]
            for number in range(len(classification_data_bin)-1):
                val_i = int(classification_data_perm[number])
                val_j = int(classification_data_perm[number+1])
                transition_matrix_veh_random_[n,bin,val_i,val_j]+=1

for bin in range(0,12):
    for i in range(3):
        if np.sum(transition_matrix_veh[bin,i,:]):
            transition_matrix_veh[bin,i,:] = transition_matrix_veh[bin,i,:]/np.sum(transition_matrix_veh[bin,i,:])
for n in range(N_perm):
    for bin in range(0, 12):
        for i in range(3):
            if np.sum(transition_matrix_veh_random_[n,bin,i,:]):
                transition_matrix_veh_random_[n,bin,i,:] = transition_matrix_veh_random_[n,bin,i,:]/np.sum(transition_matrix_veh_random_[n,bin,i,:])

transition_matrix_cbd = np.zeros((12,3,3))
transition_matrix_cbd_random_ = np.zeros((N_perm,12,3,3))

for rat in range(len(rat_ID_cbd)):
    index = np.where(rat_vector_cbd == rat)[0]
    time_data = time_vector_cbd[index]
    sorted_index = np.argsort(time_data)
    time_data = time_data[sorted_index]
    classification_data = class_vector_cbd[index]
    classification_data = classification_data[sorted_index]
    bin_data = bin_vector_cbd[index]
    bin_data = bin_data[sorted_index]
    for bin in range(0,12):
        bin_index = np.where(bin_data==bin)[0]
        classification_data_bin = classification_data[bin_index]
        for number in range(len(classification_data_bin)-1):
            val_i = int(classification_data_bin[number])
            val_j = int(classification_data_bin[number+1])
            transition_matrix_cbd[bin,val_i,val_j]+=1
        for n in range(N_perm):
            permutation = np.random.permutation(len(classification_data_bin))
            classification_data_perm = classification_data_bin[permutation]
            for number in range(len(classification_data_bin) - 1):
                val_i = int(classification_data_perm[number])
                val_j = int(classification_data_perm[number + 1])
                transition_matrix_cbd_random_[n, bin, val_i, val_j] += 1

for bin in range(0,12):
    for i in range(3):
        transition_matrix_cbd[bin,i,:] = transition_matrix_cbd[bin,i,:]/np.sum(transition_matrix_cbd[bin,i,:])

for n in range(N_perm):
    for bin in range(0, 12):
        for i in range(3):
            if np.sum(transition_matrix_cbd_random_[n,bin,i,:]):
                transition_matrix_cbd_random_[n,bin,i,:] = transition_matrix_cbd_random_[n,bin,i,:]/np.sum(transition_matrix_cbd_random_[n,bin,i,:])

transition_matrix_veh_random = np.mean(transition_matrix_veh_random_,axis=0)
transition_matrix_veh_random_std = np.std(transition_matrix_veh_random_,axis=0)
transition_matrix_cbd_random = np.mean(transition_matrix_cbd_random_,axis=0)
transition_matrix_cbd_random_std = np.std(transition_matrix_cbd_random_,axis=0)

ploting_bins = np.arange(1,10)

transition_legend = ['CSWR-CSWR','R-R','SWR-SWR']
figure, axes = plt.subplots(2,1)

axes[0].plot(ploting_bins+1,transition_matrix_veh[ploting_bins,0,0],c = 'orange',marker = 'o',markersize = 3)
axes[0].plot(ploting_bins+1,transition_matrix_veh[ploting_bins,1,1],c = 'b',marker = 'o',markersize = 3)
axes[0].plot(ploting_bins+1,transition_matrix_veh[ploting_bins,2,2], c = 'r',marker = 'o',markersize = 3)
axes[0].set_title('VEHICLE', fontsize = 20)

axes[1].plot(ploting_bins+1,transition_matrix_cbd[ploting_bins,0,0],c = 'orange',marker = '*',markersize = 3)
axes[1].plot(ploting_bins+1,transition_matrix_cbd[ploting_bins,1,1],c = 'b',marker = '*',markersize = 3)
axes[1].plot(ploting_bins+1,transition_matrix_cbd[ploting_bins,2,2], c = 'r',marker = '*',markersize = 3)
axes[1].set_title('CBD', fontsize = 20)


axes[0].fill_between(ploting_bins+1,transition_matrix_veh_random[ploting_bins,0,0]-transition_matrix_veh_random_std[ploting_bins,0,0],transition_matrix_veh_random[ploting_bins,0,0]+transition_matrix_veh_random_std[ploting_bins,0,0],color = 'gray',edgecolor='orange',alpha = 0.2)
axes[0].fill_between(ploting_bins+1,transition_matrix_veh_random[ploting_bins,1,1]-transition_matrix_veh_random_std[ploting_bins,1,1],transition_matrix_veh_random[ploting_bins,1,1]+transition_matrix_veh_random_std[ploting_bins,1,1],color = 'gray',edgecolor= 'b',alpha = 0.2)
axes[0].fill_between(ploting_bins+1,transition_matrix_veh_random[ploting_bins,2,2]-transition_matrix_veh_random_std[ploting_bins,2,2],transition_matrix_veh_random[ploting_bins,2,2]+transition_matrix_veh_random_std[ploting_bins,2,2], color = 'gray',edgecolor='r',alpha = 0.2)

axes[1].fill_between(ploting_bins+1,transition_matrix_cbd_random[ploting_bins,0,0]-transition_matrix_cbd_random_std[ploting_bins,0,0],transition_matrix_cbd_random[ploting_bins,0,0]+transition_matrix_cbd_random_std[ploting_bins,0,0],color = 'gray',edgecolor='orange',alpha = 0.2)
axes[1].fill_between(ploting_bins+1,transition_matrix_cbd_random[ploting_bins,1,1]-transition_matrix_cbd_random_std[ploting_bins,1,1],transition_matrix_cbd_random[ploting_bins,1,1]+transition_matrix_cbd_random_std[ploting_bins,1,1],color = 'gray',edgecolor='b',alpha = 0.2)
axes[1].fill_between(ploting_bins+1,transition_matrix_cbd_random[ploting_bins,2,2]-transition_matrix_cbd_random_std[ploting_bins,2,2],transition_matrix_cbd_random[ploting_bins,2,2]+transition_matrix_cbd_random_std[ploting_bins,2,2], color = 'gray',edgecolor='r',alpha = 0.2)

for i in range(2):
    axes[i].set_ylim([0,1])
    axes[i].set_xlabel('BIN NUMBER', fontsize = 15)
    axes[i].set_ylabel('T-Prob', fontsize = 15)
    axes[i].legend(transition_legend,fontsize = 15)

figure.set_size_inches([10,12])
figure.savefig(figure_path + 'ripple_transition_matrix_bins_test.png')

plt.show()

transition_legend_list = []
transition_legend_list.append(['CSWR-CSWR','CSWR-R','CSWR-SWR'])#,'CBD CSWR-CSWR','CBD CSWR-R','CBD CSWR-SWR'])
transition_legend_list.append(['R-CSWR','R-R','SWR-SWR'])#,'CBD R-CSWR','CBD R-R','CBD SWR-SWR'])
transition_legend_list.append(['SWR-CSWR','SWR-R','SWR-SWR'])#,'CBD SWR-CSWR','CBD SWR-R','CBD SWR-SWR'])

titles = ['CSWR','R','SWR']

figure, axes = plt.subplots(2,3)

for ripple_type in range(3):
    axes[0,ripple_type].plot(ploting_bins+1,transition_matrix_veh[ploting_bins,ripple_type,0],c = 'orange',marker = 'o',markersize = 3)
    axes[0,ripple_type].plot(ploting_bins+1,transition_matrix_veh[ploting_bins,ripple_type,1],c = 'b',marker = 'o',markersize = 3)
    axes[0,ripple_type].plot(ploting_bins+1,transition_matrix_veh[ploting_bins,ripple_type,2], c = 'r',marker = 'o',markersize = 3)
    axes[0,ripple_type].set_title('VEHICLE', fontsize = 20)

    axes[1,ripple_type].plot(ploting_bins+1,transition_matrix_cbd[ploting_bins,ripple_type,0],c = 'orange',marker = '*',markersize = 3,alpha = 0.5)
    axes[1,ripple_type].plot(ploting_bins+1,transition_matrix_cbd[ploting_bins,ripple_type,1],c = 'b',marker = '*',markersize = 3,alpha=0.5)
    axes[1,ripple_type].plot(ploting_bins+1,transition_matrix_cbd[ploting_bins,ripple_type,2], c = 'r',marker = '*',markersize = 3,alpha=0.5)
    axes[1,ripple_type].set_title(titles[ripple_type], fontsize = 20)

    axes[0,ripple_type].fill_between(ploting_bins + 1,
                         transition_matrix_veh_random[ploting_bins, ripple_type, 0] - transition_matrix_veh_random_std[
                             ploting_bins, ripple_type, 0],
                         transition_matrix_veh_random[ploting_bins, ripple_type, 0] + transition_matrix_veh_random_std[
                             ploting_bins, ripple_type, 0], color= 'gray', edgecolor='orange', alpha=0.2)
    axes[0,ripple_type].fill_between(ploting_bins + 1,
                         transition_matrix_veh_random[ploting_bins, ripple_type, 1] - transition_matrix_veh_random_std[
                             ploting_bins, ripple_type, 1],
                         transition_matrix_veh_random[ploting_bins, ripple_type, 1] + transition_matrix_veh_random_std[
                             ploting_bins, ripple_type, 1], color='gray',edgecolor = 'b', alpha=0.2)
    axes[0,ripple_type].fill_between(ploting_bins + 1,
                         transition_matrix_veh_random[ploting_bins, ripple_type, 2] - transition_matrix_veh_random_std[
                             ploting_bins, ripple_type, 2],
                         transition_matrix_veh_random[ploting_bins, ripple_type, 2] + transition_matrix_veh_random_std[
                             ploting_bins, ripple_type, 2], color='gray',edgecolor='r', alpha=0.2)

    axes[1,ripple_type].fill_between(ploting_bins + 1,
                         transition_matrix_cbd_random[ploting_bins, ripple_type, 0] - transition_matrix_cbd_random_std[
                             ploting_bins, ripple_type, 0],
                         transition_matrix_cbd_random[ploting_bins, ripple_type, 0] + transition_matrix_cbd_random_std[
                             ploting_bins, ripple_type, 0], color= 'gray', edgecolor='orange', alpha=0.2)
    axes[1,ripple_type].fill_between(ploting_bins + 1,
                         transition_matrix_cbd_random[ploting_bins, ripple_type, 1] - transition_matrix_cbd_random_std[
                             ploting_bins, ripple_type, 1],
                         transition_matrix_cbd_random[ploting_bins, ripple_type, 1] + transition_matrix_cbd_random_std[
                             ploting_bins, ripple_type, 1], color='gray',edgecolor = 'b', alpha=0.2)
    axes[1,ripple_type].fill_between(ploting_bins + 1,
                         transition_matrix_cbd_random[ploting_bins, ripple_type, 2] - transition_matrix_cbd_random_std[
                             ploting_bins, ripple_type, 2],
                         transition_matrix_cbd_random[ploting_bins, ripple_type, 2] + transition_matrix_cbd_random_std[
                             ploting_bins, ripple_type, 2], color='gray',edgecolor='r', alpha=0.2)

    for i in range(2):
        axes[i,ripple_type].set_ylim([0,1])
        axes[i,ripple_type].set_xlabel('BIN NUMBER', fontsize = 15)
        axes[i,ripple_type].set_ylabel('T-Prob', fontsize = 15)
        axes[i,ripple_type].legend(transition_legend_list[ripple_type],fontsize = 15)

figure.set_size_inches([20,15])
figure.savefig(figure_path + 'ripple_transition_matrix_bins_all_test.png')

plt.show()

##############################################
## transition matrix by rat

transition_matrix_veh = np.zeros((len(rat_ID_veh),12,3,3))

for rat in range(len(rat_ID_veh)):
    index = np.where(rat_vector_veh == rat)[0]
    time_data = time_vector_veh[index]
    sorted_index = np.argsort(time_data)
    time_data = time_data[sorted_index]
    classification_data = class_vector_veh[index]
    classification_data = classification_data[sorted_index]
    bin_data = bin_vector_veh[index]
    bin_data = bin_data[sorted_index]
    for bin in range(0,12):
        bin_index = np.where(bin_data==bin)[0]
        classification_data_bin = classification_data[bin_index]
        for number in range(len(classification_data_bin)-1):
            val_i = int(classification_data_bin[number])
            val_j = int(classification_data_bin[number+1])
            transition_matrix_veh[rat,bin,val_i,val_j]+=1

    for bin in range(0,12):
        for i in range(3):
            transition_matrix_veh[rat,bin,i,:] = transition_matrix_veh[rat,bin,i,:]/np.sum(transition_matrix_veh[rat,bin,i,:])

transition_matrix_cbd = np.zeros((len(rat_ID_cbd),12,3,3))

for rat in range(len(rat_ID_cbd)):
    index = np.where(rat_vector_cbd == rat)[0]
    time_data = time_vector_cbd[index]
    sorted_index = np.argsort(time_data)
    time_data = time_data[sorted_index]
    classification_data = class_vector_cbd[index]
    classification_data = classification_data[sorted_index]
    bin_data = bin_vector_cbd[index]
    bin_data = bin_data[sorted_index]
    for bin in range(0,12):
        bin_index = np.where(bin_data==bin)[0]
        classification_data_bin = classification_data[bin_index]
        for number in range(len(classification_data_bin)-1):
            val_i = int(classification_data_bin[number])
            val_j = int(classification_data_bin[number+1])
            transition_matrix_cbd[rat,bin,val_i,val_j]+=1

    for bin in range(0,12):
        for i in range(3):
            transition_matrix_cbd[rat,bin,i,:] = transition_matrix_cbd[rat,bin,i,:]/np.sum(transition_matrix_cbd[rat,bin,i,:])
