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

count = 0
for type in range(8):
    for treat in range(2):
        input_file_path = data_path_features + files_prefix[type] + treatment[treat] + '.xlsx'
        data = pd.read_excel(input_file_path)
        data_array = data.to_numpy()
        count +=data.shape[0]
data_matrix = np.zeros((count,data.shape[1] + 5))
count = 0
for type in range(8):
    for treat in range(2):
        input_file_path = data_path_features + files_prefix[type] + treatment[treat] + '.xlsx'
        data = pd.read_excel(input_file_path)
        data_array = data.to_numpy()
        data_matrix[count:count+data.shape[0],0:data.shape[1]] = data_array
        data_matrix[count:count+data.shape[0],data.shape[1]] = np.ones((data.shape[0],)) * int((type % 4))
        data_matrix[count:count+data.shape[0],data.shape[1] + 1] = np.ones((data.shape[0],)) * treat
        layer = int(type/4)
        data_matrix[count:count+data.shape[0],data.shape[1]+2] = np.ones((data.shape[0],)) * layer
        data_matrix[count:count+data.shape[0],data.shape[1]+3] = np.ones((data.shape[0],)) * 0
        if type == 1 or type == 4:
            data_matrix[count:count + data.shape[0], data.shape[1] + 3] = np.ones((data.shape[0],))
        data_matrix[count:count+data.shape[0],data.shape[1]+4] = np.ones((data.shape[0],)) * type
        count +=data.shape[0]

data_pd = pd.DataFrame(data_matrix.T, features).transpose()
data_pd.to_csv(data_path_features + 'features_df.csv')