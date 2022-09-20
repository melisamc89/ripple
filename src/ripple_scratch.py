
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle


#rat_ID = [3,4,9,201,203,206,210,211,213]
rat_ID = [2,5,10,11,204,205,207,209,212,214]
keywords = ['HPCpyra_complex_swr_veh','HPCpyra_ripple_veh','HPCpyra_swr_veh']
keywords = ['HPCpyra_complex_swr_cbd','HPCpyra_ripple_cbd','HPCpyra_swr_cbd']

type_label = ['ComplexRipple','Ripple','SWR']
srate = 600
rat_number = 0

data_path = '/home/melisamc/Documentos/ripple/data/CBD_PCA_HPCpyra/'
figure_path = '/home/melisamc/Documentos/ripple/figures/cbd_filter/'

color_list = ['b','r','g']
time = np.arange(1500, 2000)
time2 = np.arange(225, 375)

mean_matrix_rats = np.zeros((len(rat_ID),3, time.shape[0]))

for rat_number in range(len(rat_ID)):

    #data_file = 'HPCpyra_events_ratID' + str(rat_ID[rat_number]) + '.mat'
    data_file = 'PCA_HPCpyra_events_ratID' + str(rat_ID[rat_number]) + '.mat'
    data = sio.loadmat(data_path + data_file)


    figure = plt.figure()
    gs = plt.GridSpec(12, 2)

    axes2 = figure.add_subplot(gs[6:8, 1], projection = '3d')
    axes2_ = figure.add_subplot(gs[10:12, 0:2])


    mean_matrix = np.zeros((3,time.shape[0]))

    for type1 in range(2):
        for type2 in range(2):
            index = type1*2+type2
            if index < 3:
                axes1 = figure.add_subplot(gs[type1*5:type1*5+3, type2])
                axes1_ = figure.add_subplot(gs[type1*5+3:type1*5+4, type2])

                ripple_type = data[keywords[index]]
                for i in range(ripple_type.shape[0]):
                    if i < 50:
                        norm_ripple = (ripple_type[i,:]-np.min(ripple_type[i,:]))/np.max(ripple_type[i,:])
                        axes1.plot(time/srate - 3,norm_ripple[time] + i)

                mean_ripple = np.mean(ripple_type[:,time],axis=0)
                axes1_.plot(time/srate - 3,mean_ripple, color = color_list[index])
                axes2_.plot(time[time2]/srate - 3,mean_ripple[time2], color = color_list[index])
                mean_matrix[index,:] = mean_ripple
                mean_matrix_rats[rat_number,index,:] = mean_ripple
                axes1_.set_xlabel('Time [s]', fontsize = 20)
                axes1.set_xticklabels([])
                axes1.set_ylabel(type_label[index], fontsize = 20)

    axes2.scatter(mean_matrix[0,time2],mean_matrix[1,time2],mean_matrix[2,time2], color = 'k')
    axes2.set_xlabel(type_label[0],fontsize = 12)
    axes2.set_ylabel(type_label[1],fontsize = 12)
    axes2.set_zlabel(type_label[2],fontsize = 12)

    axes2_.set_xlabel('Time [s]', fontsize=20)
    axes2_.legend(type_label, fontsize = 15)
    figure.suptitle('Ripples Rat ' + str(rat_ID[rat_number]), fontsize = 25)
    figure.set_size_inches([15,25])
    figure.savefig(figure_path + 'ripples_types_RAT_' + str(rat_ID[rat_number])+'.png')
    plt.show()


mean_ripple_matrix = np.mean(mean_matrix_rats,axis = 0)
std_ripple_matrix = np.std(mean_matrix_rats,axis = 0)

two_class_matrix1 = np.zeros((2,mean_ripple_matrix.shape[1]))
two_class_matrix1[0,:] = mean_ripple_matrix[0,:]
two_class_matrix1[1,:] = mean_ripple_matrix[1,:]

two_class_matrix2 = np.zeros((2,mean_ripple_matrix.shape[1]))
two_class_matrix2[0,:] = mean_ripple_matrix[0,:]
two_class_matrix2[1,:] = mean_ripple_matrix[2,:]

mean_ripple_matrix1 = np.zeros((2,mean_ripple_matrix.shape[1]))
mean_ripple_matrix1[0,:] = np.mean(two_class_matrix1,axis = 0)
mean_ripple_matrix1[1,:] = mean_ripple_matrix[2,:]

std_ripple_matrix1 = np.zeros((2,mean_ripple_matrix.shape[1]))
std_ripple_matrix1[0,:] = np.std(two_class_matrix1,axis = 0)
std_ripple_matrix1[1,:] = std_ripple_matrix[2,:]

mean_ripple_matrix2 = np.zeros((2,mean_ripple_matrix.shape[1]))
mean_ripple_matrix2[0,:] = np.mean(two_class_matrix2,axis = 0)
mean_ripple_matrix2[1,:] = mean_ripple_matrix[1,:]

std_ripple_matrix2 = np.zeros((2,mean_ripple_matrix.shape[1]))
std_ripple_matrix2[0,:] = np.std(two_class_matrix2,axis = 0)
std_ripple_matrix2[1,:] = std_ripple_matrix[1,:]


figure, axes = plt.subplots()

# for i in range(3):
#     for rat in range(len(rat_ID)):
#         axes.plot(time[time2]/srate - 3,mean_matrix_rats[rat,i, time2], color=color_list[i],alpha=0.2)
# for i in range(3):
#     axes.plot(time[time2]/srate-3,mean_ripple_matrix[i,time2],color=color_list[i])

for i in range(3):
    axes.plot(time[time2]/srate-3,mean_ripple_matrix[i,time2],color=color_list[i])
for i in range(3):
    axes.fill_between(time[time2]/srate-3,mean_ripple_matrix[i,time2]-std_ripple_matrix[i,time2],mean_ripple_matrix[i,time2]+std_ripple_matrix[i,time2],color=color_list[i],alpha=0.3)

axes.set_xlabel('Time[s]', fontsize = 15)
axes.set_ylabel('Activity [units?]', fontsize = 15)
axes.legend(type_label, fontsize=15)
figure.savefig(figure_path + 'ripples_types_mean_2.png')

plt.show()

figure, axes = plt.subplots()
for i in range(2):
    axes.plot(time[time2]/srate-3,mean_ripple_matrix1[i,time2],color=color_list[i])
for i in range(2):
    axes.fill_between(time[time2]/srate-3,mean_ripple_matrix1[i,time2]-std_ripple_matrix1[i,time2],mean_ripple_matrix1[i,time2]+std_ripple_matrix1[i,time2],color=color_list[i],alpha=0.3)
axes.set_xlabel('Time[s]', fontsize = 15)
axes.set_ylabel('Activity [units?]', fontsize = 15)
axes.legend(['ComplexSWR+Ripple','SWR'], fontsize=15)
figure.savefig(figure_path + 'ripples_mean_binary_1.png')

plt.show()

figure, axes = plt.subplots()
for i in range(2):
    axes.plot(time[time2]/srate-3,mean_ripple_matrix2[i,time2],color=color_list[i])
for i in range(2):
    axes.fill_between(time[time2]/srate-3,mean_ripple_matrix2[i,time2]-std_ripple_matrix2[i,time2],mean_ripple_matrix2[i,time2]+std_ripple_matrix2[i,time2],color=color_list[i],alpha=0.3)
axes.set_xlabel('Time[s]', fontsize = 15)
axes.set_ylabel('Activity [units?]', fontsize = 15)
axes.legend(['ComplexSWR+SWR','Ripple'], fontsize=15)
figure.savefig(figure_path + 'ripples_mean_binary_2.png')

plt.show()

figure, axes = plt.subplots(3,1)

for i in range(3):
    for rat in range(len(rat_ID)):
        axes[i].plot(time[time2]/srate - 3,mean_matrix_rats[rat,i, time2], color=color_list[i],alpha=0.2)
    axes[i].set_ylabel('Mean Act', fontsize = 5)
    axes[i].set_title(type_label[i],fontsize= 20)

axes[2].set_xlabel('Time[s]', fontsize = 15)

figure.savefig(figure_path + 'ripples_types_mean_rats.png')
figure.set_size_inches([25, 15])


plt.show()

figure, axes = plt.subplots(2,1)
label = ['ripple','swr']
for i in range(3):
    for rat in range(len(rat_ID)):
        if i == 0 or i ==1:
            axes[0].plot(time[time2]/srate - 3,mean_matrix_rats[rat,i, time2], color=color_list[i],alpha=0.2)
        else:
            axes[1].plot(time[time2]/srate - 3,mean_matrix_rats[rat,i, time2], color=color_list[i],alpha=0.2)

for i in range(2):
    axes[i].set_ylabel('Mean Act', fontsize = 5)
    axes[i].set_title(label[i],fontsize= 20)

axes[1].set_xlabel('Time[s]', fontsize = 15)

figure.savefig(figure_path + 'ripples_types_mean_rats_ripple_vs_swr.png')
figure.set_size_inches([25, 15])

plt.show()


figure, axes = plt.subplots(2,1)
label = ['ripple','swr']
for i in range(3):
    for rat in range(len(rat_ID)):
        if i == 1:
            axes[0].plot(time[time2]/srate - 3,mean_matrix_rats[rat,i, time2], color=color_list[i],alpha=0.2)
        else:
            axes[1].plot(time[time2]/srate - 3,mean_matrix_rats[rat,i, time2], color=color_list[i],alpha=0.2)

for i in range(2):
    axes[i].set_ylabel('Mean Act', fontsize = 5)
    axes[i].set_title(label[i],fontsize= 20)

axes[1].set_xlabel('Time[s]', fontsize = 15)

figure.savefig(figure_path + 'ripples_types_mean_rats_sw_vs_ripple.png')
figure.set_size_inches([25, 15])

plt.show()