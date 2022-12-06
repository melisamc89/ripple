import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from scipy import signal
from scipy.io import savemat
import pandas as pd
from scipy import stats

data_path = '/home/melisamc/Documentos/ripple/data/features_meli/'
figure_path = '/home/melisamc/Documentos/ripple/figures/'

rat_ID_veh = [3,4,9,201,203,206,210,211,213]
rat_ID_cbd= [2,5,10,11,204,205,207,209,212,214]


counter = 0
for rat_number in range(len(rat_ID_veh)):
    data_file = 'GC_ratID' + str(rat_ID_veh[rat_number]) + '_veh_features.npy'
    data = np.load(data_path + data_file)
    counter += data.shape[0]

all_data = np.zeros((counter,10))
counter = 0
for rat_number in range(len(rat_ID_veh)):
    data_file = 'GC_ratID' + str(rat_ID_veh[rat_number]) + '_veh_features.npy'
    data = np.load(data_path + data_file)
    all_data[counter:counter+data.shape[0],:] = data
    counter += data.shape[0]
pd.DataFrame(all_data).to_csv(data_path + 'features_veh.csv')

counter = 0
for rat_number in range(len(rat_ID_cbd)-1):
    data_file = 'GC_ratID' + str(rat_ID_cbd[rat_number]) + '_cbd_features.npy'
    data = np.load(data_path + data_file)
    counter += data.shape[0]

all_data_cbd = np.zeros((counter,10))
counter = 0
for rat_number in range(len(rat_ID_cbd)-1):
    data_file = 'GC_ratID' + str(rat_ID_cbd[rat_number]) + '_cbd_features.npy'
    data = np.load(data_path + data_file)
    all_data_cbd[counter:counter+data.shape[0],:] = data
    counter += data.shape[0]
pd.DataFrame(all_data_cbd).to_csv(data_path + 'features_cbd.csv')


##### S-K TEST#############

### SW
index = np.where(all_data[:,9] == 0)[0]
sample1= np.sort(all_data[index,6])
index = np.where(all_data[:,9] == 2)[0]
sample2 = np.sort(all_data[index,6])
index = np.where(all_data[:,9] == 3)[0]
sample3 = np.sort(all_data[index,6])
#
############################


multi_factor = 1/600
#### frequency distributions
figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,2])
axes.plot(feature.T,number.T, 'r')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,2])
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,2])
axes.plot(feature.T,number.T,'yellow')
axes.set_xlim([0,20])
axes.set_ylabel('P ( X > ' + 'Freq' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Freq',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_'+'SW'+'_'+'freq'+'.pdf',bbox_inches='tight')

figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,3])
axes.plot(feature.T,number.T, 'b')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,3])
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,3])
axes.plot(feature.T,number.T,'yellow')
axes.set_xlim([100,160])
axes.set_ylabel('P ( X > ' + 'Freq' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Freq',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_'+'R'+'_'+'freq'+'.pdf',bbox_inches='tight')

### Envelope plot
figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,4])*multi_factor
axes.plot(feature.T,number.T, 'r')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,4])*multi_factor
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,4])*multi_factor
axes.plot(feature.T,number.T,'yellow')
axes.set_xscale('log')
axes.set_xlim([0.1,1000])
axes.set_ylabel('P ( X > ' + 'Env' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Env',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_'+'SW'+'_'+'envelope'+'.pdf',bbox_inches='tight')

figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,5])*multi_factor
axes.plot(feature.T,number.T, 'b')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,5])*multi_factor
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,5])*multi_factor
axes.plot(feature.T,number.T,'yellow')
axes.set_xscale('log')
axes.set_xlim([0.1,100])
axes.set_ylabel('P ( X > ' + 'Env' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Env',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_'+'R'+'_'+'envelope'+'.pdf',bbox_inches='tight')

### AUC plot
figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,6])*multi_factor
axes.plot(feature.T,number.T, 'r')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,6])*multi_factor
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,6])*multi_factor
axes.plot(feature.T,number.T,'yellow')
axes.set_xscale('log')
axes.set_xlim([0.1,1000])
axes.set_ylabel('P ( X > ' + 'AUC' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('AUC',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_'+'SW'+'_'+'AUC'+'.pdf',bbox_inches='tight')

figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,7])*multi_factor
axes.plot(feature.T,number.T, 'b')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,7])*multi_factor
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,7])*multi_factor
axes.plot(feature.T,number.T,'yellow')
axes.set_xscale('log')
axes.set_xlim([0.1,100])
axes.set_ylabel('P ( X > ' + 'AUC' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('AUC',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_'+'R'+'_'+'AUC'+'.pdf',bbox_inches='tight')

### Duration plot
figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T, 'r')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T,'yellow')
axes.set_xscale('log')
axes.set_xlim([1,1000])
axes.set_ylabel('P ( X > ' + 'Duration' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Duration',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_'+'SW'+'_'+'Duration'+'.pdf',bbox_inches='tight')


figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T, 'b')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T,'yellow')
axes.set_xlim([1,1000])
axes.set_xscale('log')
axes.set_ylabel('P ( X > ' + 'Duration' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Duration',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_'+'R'+'_'+'Duration'+'.pdf',bbox_inches='tight')


#### AUC vs AUC
figure, axes = plt.subplots()
color_list = ['r','b','purple','yellow']
for i in range(4):
    index = np.where(all_data[:,9] == i)[0]
    axes.scatter(all_data[index,6]*multi_factor,all_data[index,7]*multi_factor, color = color_list[i],alpha = 0.2)
axes.set_xlabel('AUC SR',fontsize = 10,fontname="Arial")
axes.set_ylabel('AUC Pyr',fontsize = 10,fontname="Arial")
#axes.set_xscale('log')
#axes.set_yscale('log')
axes.tick_params(axis='both', which='major', labelsize=10)
#figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_AUC_lin.pdf',bbox_inches='tight')


#### ENvelope vs Envelope
figure, axes = plt.subplots()
color_list = ['r','b','purple','yellow']
for i in range(4):
    index = np.where(all_data[:,9] == i)[0]
    axes.scatter(all_data[index,4]*multi_factor,all_data[index,5]*multi_factor, color = color_list[i],alpha = 0.2)
axes.set_xlabel('ENV SR',fontsize = 10,fontname="Arial")
axes.set_ylabel('ENV Pyr',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
axes.set_ylim([0,40])
axes.set_xlim([0,600])
#figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_ENV.pdf',bbox_inches='tight')

### Freq vs Freq
figure, axes = plt.subplots()
color_list = ['r','b','purple','yellow']
for i in range(4):
    index = np.where(all_data[:,9] == i)[0]
    axes.scatter(all_data[index,2],all_data[index,3], color = color_list[i],alpha = 0.2)
axes.set_xlabel('Freq SR',fontsize = 10,fontname="Arial")
axes.set_ylabel('Freq Pyr',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
#figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_Freq.pdf',bbox_inches='tight')

### Freq vs Duration plot
figure, axes = plt.subplots()
color_list = ['r','b','purple','yellow']
for i in [0,2,3]:
    index = np.where(all_data[:,9] == i)[0]
    axes.scatter(all_data[index,2],all_data[index,8], color = color_list[i],alpha = 0.2)
axes.set_xlabel('Freq SR',fontsize = 10,fontname="Arial")
axes.set_ylabel('Duration SW',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
#figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_Freq_vs_Duration.pdf',bbox_inches='tight')

### Freq vs Duration plot
figure, axes = plt.subplots()
color_list = ['r','b','purple','yellow']
for i in [1,2,3]:
    index = np.where(all_data[:,9] == i)[0]
    axes.scatter(all_data[index,3],all_data[index,8], color = color_list[i],alpha = 0.2)
axes.set_xlabel('Freq R',fontsize = 10,fontname="Arial")
axes.set_ylabel('Duration R',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
#figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_Freq_vs_Duration_R.pdf',bbox_inches='tight')


#### Combined VEh + CBD

figure, axes = plt.subplots()
index = np.where(all_data[:,9] >= 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T, 'k')
index = np.where(all_data_cbd[:,9] >= 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data_cbd[index,8])
axes.plot(feature.T,number.T,'g')
axes.set_xscale('log')
axes.set_xlim([1,1000])
axes.set_ylabel('P ( X > ' + 'Duration' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Duration',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_Duration.pdf',bbox_inches='tight')



figure, axes = plt.subplots()
index = np.where(all_data[:,9] > 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T, 'k')
index = np.where(all_data_cbd[:,9] > 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data_cbd[index,8])
axes.plot(feature.T,number.T,'g')
axes.set_xscale('log')
axes.set_xlim([1,1000])
axes.set_ylabel('P ( X > ' + 'Duration' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Duration',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_pyr_Duration_all.pdf',bbox_inches='tight')

figure, axes = plt.subplots()
index = np.where(all_data[:,9] != 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T, 'k')
index = np.where(all_data_cbd[:,9] != 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data_cbd[index,8])
axes.plot(feature.T,number.T,'g')
axes.set_xscale('log')
axes.set_xlim([1,1000])
axes.set_ylabel('P ( X > ' + 'Duration' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Duration',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_rad_Duration_all.pdf',bbox_inches='tight')


figure, axes = plt.subplots()
index = np.where(all_data[:,9] >= 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,2])
axes.plot(feature.T,number.T, 'k')
index = np.where(all_data_cbd[:,9] >= 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data_cbd[index,2])
axes.plot(feature.T,number.T,'g')
#axes.set_xscale('log')
axes.set_xlim([0,20])
axes.set_ylabel('P ( X > ' + 'Freq' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Freq',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_Frequency_SW.pdf',bbox_inches='tight')

figure, axes = plt.subplots()
index = np.where(all_data[:,9] >= 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,3])
axes.plot(feature.T,number.T, 'k')
index = np.where(all_data_cbd[:,9] >= 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data_cbd[index,3])
axes.plot(feature.T,number.T,'g')
#axes.set_xscale('log')
axes.set_xlim([100,160])
axes.set_ylabel('P ( X > ' + 'Freq' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Freq',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_Frequency_R.pdf',bbox_inches='tight')


figure, axes = plt.subplots()
index = np.where(all_data[:,9] != 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,6])*multi_factor
axes.plot(feature.T,number.T, 'k')
index = np.where(all_data_cbd[:,9] != 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data_cbd[index,6])*multi_factor
axes.plot(feature.T,number.T,'g')
axes.set_xscale('log')
axes.set_xlim([0.1,1000])
axes.set_ylabel('P ( X > ' + 'AUC' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('AUC',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_AUC_SW.pdf',bbox_inches='tight')

figure, axes = plt.subplots()
index = np.where(all_data[:,9] != 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,7])*multi_factor
axes.plot(feature.T,number.T, 'k')
index = np.where(all_data_cbd[:,9] != 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data_cbd[index,7])*multi_factor
axes.plot(feature.T,number.T,'g')
axes.set_xscale('log')
axes.set_xlim([0.1,100])
axes.set_ylabel('P ( X > ' + 'AUC' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('AUC',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_AUC_R.pdf',bbox_inches='tight')



figure, axes = plt.subplots()
index = np.where(all_data[:,9] != 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,4])*multi_factor
axes.plot(feature.T,number.T, 'k')
index = np.where(all_data_cbd[:,9] != 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data_cbd[index,4])*multi_factor
axes.plot(feature.T,number.T,'g')
axes.set_xscale('log')
axes.set_xlim([0.1,1000])
axes.set_ylabel('P ( X > ' + 'Env' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Env',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_env_SW.pdf',bbox_inches='tight')

figure, axes = plt.subplots()
index = np.where(all_data[:,9] != 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,5])*multi_factor
axes.plot(feature.T,number.T, 'k')
index = np.where(all_data_cbd[:,9] != 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data_cbd[index,5])*multi_factor
axes.plot(feature.T,number.T,'g')
axes.set_xscale('log')
axes.set_xlim([0.1,100])
axes.set_ylabel('P ( X > ' + 'Env' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Env',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_env_R.pdf',bbox_inches='tight')


########################################################

all_data = all_data_cbd.copy()
#### frequency distributions
figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,2])
axes.plot(feature.T,number.T, 'r')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,2])
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,2])
axes.plot(feature.T,number.T,'yellow')
axes.set_xlim([0,20])
axes.set_ylabel('P ( X > ' + 'Freq' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Freq',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_cbd_'+'SW'+'_'+'freq'+'.pdf',bbox_inches='tight')

figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,3])
axes.plot(feature.T,number.T, 'b')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,3])
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,3])
axes.plot(feature.T,number.T,'yellow')
axes.set_xlim([100,160])
axes.set_ylabel('P ( X > ' + 'Freq' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Freq',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_cbd_'+'R'+'_'+'freq'+'.pdf',bbox_inches='tight')

### Envelope plot
figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,4])*multi_factor
axes.plot(feature.T,number.T, 'r')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,4])*multi_factor
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,4])*multi_factor
axes.plot(feature.T,number.T,'yellow')
axes.set_xscale('log')
axes.set_xlim([0.1,1000])
axes.set_ylabel('P ( X > ' + 'Env' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Env',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_cbd_'+'SW'+'_'+'envelope'+'.pdf',bbox_inches='tight')

figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,5])*multi_factor
axes.plot(feature.T,number.T, 'b')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,5])*multi_factor
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,5])*multi_factor
axes.plot(feature.T,number.T,'yellow')
axes.set_xscale('log')
axes.set_xlim([0.1,100])
axes.set_ylabel('P ( X > ' + 'Env' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Env',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_cbd_'+'R'+'_'+'envelope'+'.pdf',bbox_inches='tight')

### AUC plot
figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,6])*multi_factor
axes.plot(feature.T,number.T, 'r')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,6])*multi_factor
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,6])*multi_factor
axes.plot(feature.T,number.T,'yellow')
axes.set_xscale('log')
axes.set_xlim([0.1,1000])
axes.set_ylabel('P ( X > ' + 'AUC' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('AUC',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_cbd_'+'SW'+'_'+'AUC'+'.pdf',bbox_inches='tight')

figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,7])*multi_factor
axes.plot(feature.T,number.T, 'b')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,7])*multi_factor
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,7])*multi_factor
axes.plot(feature.T,number.T,'yellow')
axes.set_xscale('log')
axes.set_xlim([0.1,100])

axes.set_ylabel('P ( X > ' + 'AUC' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('AUC',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_cbd_'+'R'+'_'+'AUC'+'.pdf',bbox_inches='tight')

### Duration plot
figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 0)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T, 'r')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T,'yellow')
axes.set_xscale('log')
axes.set_xlim([1,1000])
axes.set_ylabel('P ( X > ' + 'Duration' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Duration',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_cbd_'+'SW'+'_'+'Duration'+'.pdf',bbox_inches='tight')


figure, axes = plt.subplots()
index = np.where(all_data[:,9] == 1)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T, 'b')
index = np.where(all_data[:,9] == 2)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T,'purple')
index = np.where(all_data[:,9] == 3)[0]
number = 1 - np.arange(0,index.shape[0])/len(index)
feature = np.sort(all_data[index,8])
axes.plot(feature.T,number.T,'yellow')
axes.set_xlim([1,1000])
axes.set_xscale('log')
axes.set_ylabel('P ( X > ' + 'Duration' + ')',fontsize = 10,fontname="Arial")
axes.set_xlabel('Duration',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_cbd_'+'R'+'_'+'Duration'+'.pdf',bbox_inches='tight')


#### AUC vs AUC
figure, axes = plt.subplots()
color_list = ['r','b','purple','yellow']
for i in range(4):
    index = np.where(all_data[:,9] == i)[0]
    axes.scatter(all_data[index,6]*multi_factor,all_data[index,7]*multi_factor, color = color_list[i],alpha = 0.2)
axes.set_xlabel('AUC SR',fontsize = 10,fontname="Arial")
axes.set_ylabel('AUC Pyr',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
axes.set_xscale('log')
axes.set_yscale('log')

#figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_AUC_cbd.pdf',bbox_inches='tight')


#### ENvelope vs Envelope
figure, axes = plt.subplots()
color_list = ['r','b','purple','yellow']
for i in range(4):
    index = np.where(all_data[:,9] == i)[0]
    axes.scatter(all_data[index,4]*multi_factor,all_data[index,5]*multi_factor, color = color_list[i],alpha = 0.2)

axes.set_xlabel('ENV SR',fontsize = 10,fontname="Arial")
axes.set_ylabel('ENV Pyr',fontsize = 10,fontname="Arial")
axes.set_xlim([0,600])
axes.set_ylim([0,40])
axes.tick_params(axis='both', which='major', labelsize=10)
#figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_ENV_cbd.pdf',bbox_inches='tight')

### Freq vs Freq
figure, axes = plt.subplots()
color_list = ['r','b','purple','yellow']
for i in range(4):
    index = np.where(all_data[:,9] == i)[0]
    axes.scatter(all_data[index,2],all_data[index,3], color = color_list[i],alpha = 0.2)

axes.set_xlabel('Freq SR',fontsize = 10,fontname="Arial")
axes.set_ylabel('Freq Pyr',fontsize = 10,fontname="Arial")
axes.tick_params(axis='both', which='major', labelsize=10)
#figure.set_size_inches([1.2,1.2])
figure.savefig(figure_path + 'features_Freq_cbd.pdf',bbox_inches='tight')
