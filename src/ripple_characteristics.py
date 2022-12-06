import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from scipy import signal
from scipy.io import savemat
import pandas as pd


data_path_features= '/home/melisamc/Documentos/ripple/data/characteristics/'
figure_path = '/home/melisamc/Documentos/ripple/figures/'


files_prefix = ['sw_','r_bel_','swr_','cswr_','sw_pyr_','r_','swr_pyr_','cswr_pyr_']
treatment = ['veh','cbd']
features = ['Inst_Freq','Mean_Freq','Amp','AUC','Duration','P2P','Entropy','Spec_Entropy','number','treatment','layer','belonging','classification']

data_pd = pd.read_csv(data_path_features + 'features_df.csv')

LAYER = ['s','r']

for layer in range(2):
    veh_sw = data_pd.loc[(data_pd['treatment'] ==0) & (data_pd['layer'] == layer) & (data_pd['belonging'] == 0)]
    cbd_sw = data_pd.loc[(data_pd['treatment'] ==1) & (data_pd['layer'] == layer) & (data_pd['belonging'] == 0)]
    for index in [1,3,4]:
        figure, axes = plt.subplots()
        features_veh = veh_sw[features[index]]
        features_cbd = cbd_sw[features[index]]
        order_veh = np.arange(0,features_veh.shape[0])/features_veh.shape[0]
        order_cbd = np.arange(0,features_cbd.shape[0])/features_cbd.shape[0]
        sorted_veh = np.sort(features_veh)
        sorted_cbd = np.sort(features_cbd)
        axes.scatter(sorted_veh,1-order_veh, color = 'k',s = 0.5, alpha = 0.5)
        axes.scatter(sorted_cbd,1-order_cbd,color = 'g', s = 0.5, alpha = 0.5)
        axes.set_ylabel('P ( X > ' + features[index] + ')',fontsize = 10,fontname="Arial")
        axes.set_xlabel(features[index],fontsize = 10,fontname="Arial")
        # if index != 0 and index !=1:
        #    axes.set_xscale('log')
        axes.set_yscale('log')
        axes.tick_params(axis='both', which='major', labelsize=10)
        #axes.legend(['VEH','CBD'],fontsize = 10)
        figure.set_size_inches([1.2,1.2])
        #figure.suptitle(features_label [index] + 'Cumulative Distributions', fontsize = 15)
        figure.savefig(figure_path + 'features_'+LAYER[layer]+'_'+features[index]+'.jpeg',bbox_inches='tight')
plt.show()

color_list = ['red','blue','purple','yellow']
treat = 0
treatment_label = ['veh','cbd']
for layer in range(2):
    for index in range(8):
        figure, axes = plt.subplots()
        for type in range(4):
            veh = data_pd.loc[(data_pd['treatment'] == treat) & (data_pd['layer'] == layer) & (data_pd['number'] == type) & (data_pd['belonging'] == 0)]
            features_veh = veh[features[index]]
            order_veh = np.arange(0,features_veh.shape[0])/features_veh.shape[0]
            sorted_veh = np.sort(features_veh)
            positive = np.where(sorted_veh>0)[0]
            if len(positive) :
                axes.scatter(sorted_veh[positive],1-order_veh[positive], color = color_list[type],s = 0.5, alpha = 0.5)
                axes.set_ylabel('P ( X > ' + features[index] + ')',fontsize = 10,fontname="Arial")
                axes.set_xlabel(features[index],fontsize = 10,fontname="Arial")
                # if index != 0 and index !=1:
                #    axes.set_xscale('log')
                axes.set_yscale('log')
            axes.tick_params(axis='both', which='major', labelsize=10)
            #axes.legend(['VEH','CBD'],fontsize = 10)
            figure.set_size_inches([1.2,1.2])
            #figure.suptitle(features_label [index] + 'Cumulative Distributions', fontsize = 15)
            figure.savefig(figure_path + 'features_'+LAYER[layer]+'_'+features[index]+'_'+treatment_label[treat]+'_types.jpeg',bbox_inches='tight')
plt.show()


########################3
for index in range(8):
    figure, axes = plt.subplots()
    for layer in range(2):
        veh_sw = data_pd.loc[(data_pd['treatment'] ==0) & (data_pd['layer'] == layer) & (data_pd['belonging'] == 0)]
        cbd_sw = data_pd.loc[(data_pd['treatment'] ==1) & (data_pd['layer'] == layer) & (data_pd['belonging'] == 0)]
        features_veh = veh_sw[features[index]]
        features_cbd = cbd_sw[features[index]]
        order_veh = np.arange(0,features_veh.shape[0])/features_veh.shape[0]
        order_cbd = np.arange(0,features_cbd.shape[0])/features_cbd.shape[0]
        sorted_veh = np.sort(features_veh)
        sorted_cbd = np.sort(features_cbd)
        axes.scatter(sorted_veh,1-order_veh, color = 'k',s = 0.5, alpha = 0.5)
        axes.scatter(sorted_cbd,1-order_cbd,color = 'g', s = 0.5, alpha = 0.5)
        axes.set_ylabel('P ( X > ' + features[index] + ')',fontsize = 10,fontname="Arial")
        axes.set_xlabel(features[index],fontsize = 10,fontname="Arial")
        # if index != 0 and index !=1:
        #    axes.set_xscale('log')
        axes.set_yscale('log')
        axes.tick_params(axis='both', which='major', labelsize=10)
        #axes.legend(['VEH','CBD'],fontsize = 10)
        figure.set_size_inches([1.2,1.2])
        #figure.suptitle(features_label [index] + 'Cumulative Distributions', fontsize = 15)
        figure.savefig(figure_path + 'features_'+features[index]+'.jpeg',bbox_inches='tight')
plt.show()


color_list = ['red','blue','purple','yellow']
treat = 0
treatment_label = ['veh','cbd']
for index in range(8):
    figure, axes = plt.subplots()
    for layer in range(2):
        for type in range(4):
            veh = data_pd.loc[(data_pd['treatment'] == treat) & (data_pd['layer'] == layer) & (data_pd['number'] == type) & (data_pd['belonging'] == 0)]
            features_veh = veh[features[index]]
            order_veh = np.arange(0,features_veh.shape[0])/features_veh.shape[0]
            sorted_veh = np.sort(features_veh)
            positive = np.where(sorted_veh>0)[0]
            if len(positive) :
                axes.scatter(sorted_veh[positive],1-order_veh[positive], color = color_list[type],s = 0.5, alpha = 0.5)
                axes.set_ylabel('P ( X > ' + features[index] + ')',fontsize = 10,fontname="Arial")
                axes.set_xlabel(features[index],fontsize = 10,fontname="Arial")
                # if index != 0 and index !=1:
                #    axes.set_xscale('log')
                axes.set_yscale('log')
            axes.tick_params(axis='both', which='major', labelsize=10)
            #axes.legend(['VEH','CBD'],fontsize = 10)
            figure.set_size_inches([1.2,1.2])
            #figure.suptitle(features_label [index] + 'Cumulative Distributions', fontsize = 15)
            figure.savefig(figure_path + 'features_'+features[index]+'_'+treatment_label[treat]+'_types.jpeg',bbox_inches='tight')
plt.show()

############################################################################


from sklearn.cluster import KMeans
treat = 1
veh = data_pd.loc[(data_pd['treatment'] == treat) & (data_pd['belonging'] == 0)]
matrix = np.zeros((veh.shape[0],6))
for i in range(6):
    features_veh = veh[features[i]]
    matrix[:,i] = features_veh
class_vector = veh['number']
layer_vector = veh['layer']

figure = plt.figure()
gs = plt.GridSpec(6,6)

color_list = ['red','blue','purple','yellow']
layer_mark = ['o','x']

x_lim = [180,180,3000,400,1000,4000]

for n in range(6):
    for m in range(6):
        axes1 = figure.add_subplot(gs[n, m])  # , projection='3d')
        axes1.set_xlabel(features[n], fontsize = 15)
        axes1.set_ylabel(features[m], fontsize = 15)
        for i in range(4):
            for j in range(2):
                index = np.logical_and(class_vector == i,layer_vector == j)
                if len(index):
                    axes1.scatter(matrix[index,n],matrix[index,m],marker=layer_mark[j],color = color_list[i],alpha = 0.1)
                    axes1.set_xlim([0,x_lim[n]])
                    axes1.set_ylim([0,x_lim[m]])

figure.set_size_inches([30,30])
figure.savefig(figure_path + 'features_cbd.png')
plt.show()

####################################################################3
treat = 0
veh = data_pd.loc[(data_pd['treatment'] == treat) & (data_pd['belonging'] == 0)]
matrix = np.zeros((veh.shape[0],6))
for i in range(6):
    features_veh = veh[features[i]]
    matrix[:,i] = features_veh
class_vector = veh['number']
layer_vector = veh['layer']

treat = 1
cbd = data_pd.loc[(data_pd['treatment'] == treat) & (data_pd['belonging'] == 0)]
matrix_cbd = np.zeros((cbd.shape[0],6))
for i in range(6):
    features_cbd = cbd[features[i]]
    matrix_cbd[:,i] = features_cbd
class_vector_cbd = cbd['number']
layer_vector_cbd = cbd['layer']

plots = ['SW','R','SWR','CSWR']

figure = plt.figure()
gs = plt.GridSpec(1,2)

axes1 = figure.add_subplot(gs[0, 0], projection='3d')
axes2 = figure.add_subplot(gs[0, 1], projection='3d')

for i in [0,1,2,3]:
    for j in [0,1]:
        index = np.logical_and(class_vector == i, layer_vector == j)
        index_cbd = np.logical_and(class_vector_cbd == i, layer_vector_cbd == j)
        if len(index):
            axes1.scatter(matrix[index, 1], matrix[index, 4],matrix[index,3], marker=layer_mark[j], color=color_list[i], alpha=0.1)
            axes1.set_ylim([-10,1000])
            #axes1.set_zlim([-10,400])

            axes1.set_xlabel('Mean Freq',fontsize = 20)
            axes1.set_ylabel('Duration',fontsize = 20)
            axes1.set_zlabel('AUC',fontsize = 20)

        if len(index_cbd):
            axes2.scatter(matrix_cbd[index_cbd, 1], matrix_cbd[index_cbd, 4],matrix_cbd[index_cbd,3], marker=layer_mark[j], color=color_list[i], alpha=0.1)
            axes2.set_ylim([-10,1000])
            #axes2.set_zlim([-10,400])
            axes2.set_xlabel('Mean Freq',fontsize = 20)
            axes2.set_ylabel('Duration',fontsize = 20)
            axes2.set_zlabel('AUC',fontsize= 20)


figure.set_size_inches([15,7])
figure.savefig(figure_path + 'MeanFreq_Dur_AUC_3D.png')



figure = plt.figure()
gs = plt.GridSpec(4, 2)

for i in [0,1,2,3]:
    axes1 = figure.add_subplot(gs[i, 0], projection='3d')
    axes2 = figure.add_subplot(gs[i, 1], projection='3d')
    axes1.set_title('VEHICLE', fontsize=25)
    axes2.set_title('CBD', fontsize=25)
    for j in [0,1]:
        index = np.logical_and(class_vector == i, layer_vector == j)
        index_cbd = np.logical_and(class_vector_cbd == i, layer_vector_cbd == j)
        if len(index):
            axes1.scatter(matrix[index, 1], matrix[index, 4],matrix[index,3], marker=layer_mark[j], color=color_list[i], alpha=0.1)
            axes1.set_ylim([-10,1000])
            axes1.set_zlim([-10,400])

            axes1.set_xlabel('Mean Freq',fontsize = 20)
            axes1.set_ylabel('Duration',fontsize = 20)
            axes1.set_zlabel('AUC',fontsize = 20)

        if len(index_cbd):
            axes2.scatter(matrix_cbd[index_cbd, 1], matrix_cbd[index_cbd, 4],matrix_cbd[index_cbd,3], marker=layer_mark[j], color=color_list[i], alpha=0.1)
            axes2.set_ylim([-10,1000])
            axes2.set_zlim([-10,400])
            axes2.set_xlabel('Mean Freq',fontsize = 20)
            axes2.set_ylabel('Duration',fontsize = 20)
            axes2.set_zlabel('AUC',fontsize= 20)


figure.set_size_inches([15,30])
figure.savefig(figure_path + 'MeanFreq_Dur_AUC.png')

figure = plt.figure()
gs = plt.GridSpec(1,2)

axes1 = figure.add_subplot(gs[0, 0])
axes2 = figure.add_subplot(gs[0, 1])

for i in [0,1,2,3]:
    for j in [0,1]:
        index = np.logical_and(class_vector == i, layer_vector == j)
        index_cbd = np.logical_and(class_vector_cbd == i, layer_vector_cbd == j)
        if len(index):
            axes1.scatter(matrix[index, 1], matrix[index, 4], marker=layer_mark[j], color=color_list[i], alpha=0.1)
            axes1.set_ylim([-10,1000])
            axes1.set_xlabel('Mean Freq',fontsize = 20)
            axes1.set_ylabel('Duration',fontsize = 20)

        if len(index_cbd):
            axes2.scatter(matrix_cbd[index_cbd, 1], matrix_cbd[index_cbd, 4], marker=layer_mark[j], color=color_list[i], alpha=0.1)
            axes2.set_ylim([-10,1000])
            axes2.set_xlabel('Mean Freq',fontsize = 20)
            axes2.set_ylabel('Duration',fontsize = 20)

axes1.set_title('VEHICLE', fontsize = 25)
axes2.set_title('CBD', fontsize = 25)
figure.set_size_inches([15,7])
figure.savefig(figure_path + 'MeanFreq_Dur.png')
