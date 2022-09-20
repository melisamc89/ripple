
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

###save new_rat_data
def save_pca_filter_data(input_data_path,output_data_path,ID_sequence,events_sequence,transformation):

    for rat_number in range(len(ID_sequence)):
        input_file_name = 'HPCpyra_events_ratID' + str(ID_sequence[rat_number]) + '.mat'
        output_file_name = output_data_path + 'PCA_HPCpyra_events_ratID' + str(ID_sequence[rat_number]) + '.mat'
        input_data = sio.loadmat(input_data_path + input_file_name)
        transformation_list = []
        for key in range(3):
            ripple = input_data[events_sequence[key]][:,:3600]
            x = transformation.transform(ripple)
            transform_ripple = transformation.inverse_transform(x)
            transformation_list.append(transform_ripple)
        output = {events_sequence[0]:transformation_list[0],events_sequence[1]:transformation_list[1],events_sequence[2]:transformation_list[2]}
        savemat(output_file_name,output)

    return

rat_ID_veh = [3,4,9,201,203,211,213]
rat_ID_test_veh = [206,210]
rat_ID_cbd= [2,5,10,11,204,209,212,214]
rat_ID_test_cbd = [205,207]

keywords_veh = ['HPCpyra_complex_swr_veh','HPCpyra_ripple_veh','HPCpyra_swr_veh']
keywords_cbd = ['HPCpyra_complex_swr_cbd','HPCpyra_ripple_cbd','HPCpyra_swr_cbd']

type_label = ['ComplexRipple','Ripple','SWR']
srate = 600
rat_number = 0

data_path_veh = '/home/melisamc/Documentos/ripple/data/HPCpyra/'
data_path_cbd = '/home/melisamc/Documentos/ripple/data/CBD/HPCpyra/'
output_data_path_veh = '/home/melisamc/Documentos/ripple/data/PCA_HPCpyra/'
output_data_path_cbd = '/home/melisamc/Documentos/ripple/data/CBD_PCA_HPCpyra/'

figure_path = '/home/melisamc/Documentos/ripple/figures/'

time = np.arange(0, 3600)
time2 = np.arange(0, 3600)

# time = np.arange(1500, 2000)
# time2 = np.arange(225, 375)

total_ripple_count_veh, total_matrix_veh= count_events(data_path_veh,rat_ID_veh,keywords_veh)
percetage_ripple_count_veh = total_ripple_count_veh / total_ripple_count_veh[3]
total_ripple_count_veh_test, total_matrix_test_veh= count_events(data_path_veh,rat_ID_test_veh,keywords_veh)

total_ripple_count_cbd, total_matrix_cbd = count_events(data_path_cbd,rat_ID_cbd,keywords_cbd)
percetage_ripple_count_cbd = total_ripple_count_cbd / total_ripple_count_cbd[3]
total_ripple_count_cbd_test, total_matrix_test_cbd= count_events(data_path_cbd,rat_ID_test_cbd,keywords_cbd)

ripple_veh_matrix_list, class_vector_veh= load_all_data(data_path_veh,rat_ID_veh,keywords_veh,total_ripple_count_veh,time2)
ripple_cbd_matrix_list, class_vector_cbd = load_all_data(data_path_cbd,rat_ID_cbd,keywords_cbd,total_ripple_count_cbd,time2)
ripple_veh_matrix_list_test, class_vector_veh_test = load_all_data(data_path_veh,rat_ID_test_veh,keywords_veh,total_ripple_count_veh_test,time2)
ripple_cbd_matrix_list_test, class_vector_cbd_test = load_all_data(data_path_cbd,rat_ID_test_cbd,keywords_cbd,total_ripple_count_cbd_test,time2)

number_of_components = 7
pca_training_veh = PCA(number_of_components).fit(ripple_veh_matrix_list[3])
save_pca_filter_data(data_path_veh,output_data_path_veh,rat_ID_veh,keywords_veh,pca_training_veh)
save_pca_filter_data(data_path_veh,output_data_path_veh,rat_ID_test_veh,keywords_veh,pca_training_veh)

transformed_ripple_veh_matrix_list = []
transformed_ripple_veh_matrix_list_test = []
for key in range(3):
    x = pca_training_veh.transform(ripple_veh_matrix_list[key])
    transformed_ripple_veh_matrix_list.append(pca_training_veh.inverse_transform(x))
    x = pca_training_veh.transform(ripple_veh_matrix_list_test[key])
    transformed_ripple_veh_matrix_list_test.append(pca_training_veh.inverse_transform(x))

pca_training_cbd = PCA(number_of_components).fit(ripple_cbd_matrix_list[3])
save_pca_filter_data(data_path_cbd,output_data_path_cbd,rat_ID_cbd,keywords_cbd,pca_training_cbd)
save_pca_filter_data(data_path_cbd,output_data_path_cbd,rat_ID_test_cbd,keywords_cbd,pca_training_cbd)

transformed_ripple_cbd_matrix_list = []
transformed_ripple_cbd_matrix_list_test = []
for key in range(3):
    x = pca_training_cbd.transform(ripple_cbd_matrix_list[key])
    transformed_ripple_cbd_matrix_list.append(pca_training_cbd.inverse_transform(x))
    x = pca_training_cbd.transform(ripple_cbd_matrix_list_test[key])
    transformed_ripple_cbd_matrix_list_test.append(pca_training_cbd.inverse_transform(x))

# fs = 600
# time = np.arange(1500, 2000)
# time2 = np.arange(225, 375)
# for key in range(3):
#     figure, axes = plt.subplots(1,2)
#     axes[0].set_title(type_label[key], fontsize=15)
#     axes[1].set_title('TEST Transformed', fontsize=15)
#     for i in range(50):
#         min_value = np.min(ripple_veh_matrix_list_test[key][i, time[time2]])
#         max_value = np.max(ripple_veh_matrix_list_test[key][i, time[time2]])
#         norm_version = (ripple_veh_matrix_list_test[key][i, time[time2]] - min_value) / (max_value)
#         axes[0].plot(time2 / fs, norm_version + i)
#
#         min_value = np.min(transformed_ripple_veh_matrix_list_test[key][i, time[time2]])
#         max_value = np.max(transformed_ripple_veh_matrix_list_test[key][i, time[time2]])
#         norm_version = (transformed_ripple_veh_matrix_list_test[key][i, time[time2]] - min_value) / (max_value)
#         axes[1].plot(time2 / fs, norm_version + i)
#
#     figure.savefig(figure_path + 'transformed_test_'+type_label[key]+'_'+str(number_of_components)+'.png')
#     plt.show()



output_data_path = '/home/melisamc/Documentos/ripple/data/HPCpyra_PCA_VEH_test_CBD/'
save_pca_filter_data(data_path_cbd,output_data_path,rat_ID_cbd,keywords_cbd,pca_training_veh)

total = ripple_veh_matrix_list[3].shape[0] + ripple_cbd_matrix_list[3].shape[0]
total_class = np.zeros((total,))
all_ripple = np.zeros((total,ripple_cbd_matrix_list[3].shape[1]))
all_ripple[0: ripple_veh_matrix_list[3].shape[0],:] = ripple_veh_matrix_list[3]
all_ripple[ripple_veh_matrix_list[3].shape[0]:total,:] = ripple_cbd_matrix_list[3]
total_class[0: ripple_veh_matrix_list[3].shape[0]] = class_vector_veh + np.ones_like(class_vector_veh)
total_class[ripple_veh_matrix_list[3].shape[0]:total] = (class_vector_cbd + np.ones_like(class_vector_cbd))*10


total_test = ripple_veh_matrix_list_test[3].shape[0] + ripple_cbd_matrix_list_test[3].shape[0]
total_class_test = np.zeros((total_test,))
all_ripple_test = np.zeros((total_test,ripple_cbd_matrix_list_test[3].shape[1]))
all_ripple_test[0: ripple_veh_matrix_list_test[3].shape[0],:] = ripple_veh_matrix_list_test[3]
all_ripple_test[ripple_veh_matrix_list_test[3].shape[0]:total_test,:] = ripple_cbd_matrix_list_test[3]
total_class_test[0: ripple_veh_matrix_list_test[3].shape[0]] = class_vector_veh_test + np.ones_like(class_vector_veh_test)
total_class_test[ripple_veh_matrix_list_test[3].shape[0]:total_test] = (class_vector_cdb_test + np.ones_like(class_vector_cbd_test))*10


pca_training = PCA()
pca_training.fit(all_ripple)
reduced_ripple = pca_training.transform(all_ripple)
reduced_ripple_test = pca_training.transform(all_ripple_test)
### plot
# figure = plt.figure()
# gs = plt.GridSpec(3, 3)
# limit_variance = 0.85
# time_variable = time2 / srate
# axes0 = figure.add_subplot(gs[0:3, 0])#, projection='3d')
# axes1 = figure.add_subplot(gs[0, 1])
# axes2 = figure.add_subplot(gs[1, 1])
# axes3 = figure.add_subplot(gs[2, 1])
# axes4 = figure.add_subplot(gs[0, 2])
# axes5 = figure.add_subplot(gs[1, 2])
# axes6 = figure.add_subplot(gs[2, 2])
# axes0.scatter(np.arange(0,25),pca_training.explained_variance_ratio_[0:25],color = 'k')
# exp_variance_sum = np.cumsum(pca_training.explained_variance_ratio_)
# limit = pca_training.explained_variance_ratio_[int(np.where(exp_variance_sum>limit_variance)[0][0])]
# axes0.axhline(y=limit, color='k', linestyle='-',alpha=0.5)
# axes1.plot(time_variable,pca_training.components_[0,:],color = 'k')
# axes2.plot(time_variable,pca_training.components_[1,:],color = 'k')
# axes3.plot(time_variable,pca_training.components_[2,:],color = 'k')
# axes4.plot(time_variable,pca_training.components_[3,:],color = 'k')
# axes5.plot(time_variable,pca_training.components_[4,:],color = 'k')
# axes6.plot(time_variable,pca_training.components_[5,:],color = 'k')
#
# axes0.set_xlabel('Rank',fontsize = 20)
# axes0.set_ylabel('Eigenvalue',fontsize = 20)
# axes0.set_title('Eigenspectrum',fontsize = 25)
# axes1.set_title('Eigenvectors',fontsize = 25)
#
# axes3.set_xlabel('Time (s)',fontsize = 20)
# axes6.set_xlabel('Time (s)',fontsize = 20)
#
# axes1.set_ylim([-0.5,0.5])
# axes2.set_ylim([-0.5,0.5])
# axes3.set_ylim([-0.5,0.5])
# axes4.set_ylim([-0.5,0.5])
# axes5.set_ylim([-0.5,0.5])
# axes6.set_ylim([-0.5,0.5])
#
# axes1.legend(['Direction1'])
# axes2.legend(['Direction2'])
# axes3.legend(['Direction3'])
# axes4.legend(['Direction4'])
# axes5.legend(['Direction5'])
# axes6.legend(['Direction6'])
#
# figure.set_size_inches([15,5])
# figure.savefig(figure_path+'PCA_veh_and_cbd_ripples.png')
# plt.show()
#
# ################################
color_list = ['b','r','g','cyan','magenta','yellow']
figure = plt.figure()
gs = plt.GridSpec(3, 3)
pcs = ['PC1','PC2','PC3','PC4','PC5','PC6']
for i in range(3):

    for component in range(3):
        axes0 = figure.add_subplot(gs[i, component])
        index = np.where(total_class == i+1)[0]
        axes0.scatter(reduced_ripple[index, component*2], reduced_ripple[index, component*2+1], color=color_list[i],alpha = 0.2)
        index = np.where(total_class == (i+1)*10)[0]
        axes0.scatter(reduced_ripple[index, component*2], reduced_ripple[index,  component*2+1], color=color_list[i+3],alpha = 0.2)

        axes0.set_ylim([-700,700])
        axes0.set_xlim([-700,700])
        axes0.set_xlabel(pcs[component*2] , fontsize = 15)
        axes0.set_ylabel(pcs[component*2+1] , fontsize = 15)
        if component == 0:
            axes0.set_title(type_label[i],fontsize = 25)
        axes0.legend(['VEH','CBD'], fontsize = 15)

figure.set_size_inches([20,20])
figure.savefig(figure_path + 'pca_veh_vs_cbd.png')
plt.show()

color_list = ['b','r','g','cyan','magenta','yellow']
figure = plt.figure()
gs = plt.GridSpec(3, 3)
pcs = ['PC1','PC2','PC3','PC4','PC5','PC6']
for i in range(3):

    for component in range(3):
        axes0 = figure.add_subplot(gs[i, component])
        index = np.where(total_class_test == i+1)[0]
        axes0.scatter(reduced_ripple_test[index, component*2], reduced_ripple_test[index, component*2+1], color=color_list[i],alpha = 0.2)
        index = np.where(total_class_test == (i+1)*10)[0]
        axes0.scatter(reduced_ripple_test[index, component*2], reduced_ripple_test[index,  component*2+1], color=color_list[i+3],alpha = 0.2)

        axes0.set_ylim([-700,700])
        axes0.set_xlim([-700,700])
        axes0.set_xlabel(pcs[component*2] , fontsize = 15)
        axes0.set_ylabel(pcs[component*2+1] , fontsize = 15)
        if component == 0:
            axes0.set_title(type_label[i],fontsize = 25)
        axes0.legend(['VEH','CBD'], fontsize = 15)

figure.set_size_inches([20,20])
figure.savefig(figure_path + 'pca_veh_vs_cbd_test.png')
plt.show()