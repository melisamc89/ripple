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

def create_matrix_features(data_pd, treat):

    sw = data_pd.loc[(data_pd['treatment'] == treat) & (data_pd['layer'] == 0) & (data_pd['number'] == 0) & (data_pd['belonging'] == 0)]
    sw_r = data_pd.loc[(data_pd['treatment'] == treat) & (data_pd['layer'] == 1)& (data_pd['number'] == 0) & (data_pd['belonging'] == 1)]

    r_sw = data_pd.loc[(data_pd['treatment'] == treat) & (data_pd['layer'] == 0) & (data_pd['number'] == 1) & (data_pd['belonging'] == 1)]
    r = data_pd.loc[(data_pd['treatment'] == treat) & (data_pd['layer'] == 1) & (data_pd['number'] == 1) & (data_pd['belonging'] == 0)]

    swr_sw = data_pd.loc[(data_pd['treatment'] == treat) & (data_pd['layer'] == 0) & (data_pd['number'] == 2) & (data_pd['belonging'] == 0)]
    swr_r = data_pd.loc[(data_pd['treatment'] == treat) & (data_pd['layer'] == 1) & (data_pd['number'] == 2) & (data_pd['belonging'] == 0)]

    cswr_sw = data_pd.loc[(data_pd['treatment'] == treat) & (data_pd['layer'] == 0) & (data_pd['number'] == 3) & (data_pd['belonging'] == 0)]
    cswr_r = data_pd.loc[(data_pd['treatment'] == treat) & (data_pd['layer'] == 1) & (data_pd['number'] == 3) & (data_pd['belonging'] == 0)]

    matrix_size = sw.shape[0] + r.shape[0] + swr_sw.shape[0] + cswr_sw.shape[0]
    data_matrix = np.zeros((matrix_size,17))
    count = 0
    data_matrix[0:sw.shape[0],:8] = sw.iloc[:,1:9]
    data_matrix[0:sw.shape[0],8:16] = sw_r.iloc[:,1:9]
    data_matrix[0:sw.shape[0],16] = np.zeros((sw.shape[0],))
    count += sw.shape[0]
    data_matrix[count:count + r.shape[0],:8] = r_sw.iloc[:,1:9]
    data_matrix[count:count + r.shape[0],8:16] = r.iloc[:,1:9]
    data_matrix[count: count+ r.shape[0],16] = np.ones((r.shape[0],))
    count += r.shape[0]
    data_matrix[count:count+swr_sw.shape[0],:8] = swr_sw.iloc[:,1:9]
    data_matrix[count:count+swr_sw.shape[0],8:16] = swr_r.iloc[:,1:9]
    data_matrix[count:count+swr_sw.shape[0],16] = np.ones((swr_sw.shape[0],))*2
    count += swr_sw.shape[0]
    data_matrix[count:count+cswr_sw.shape[0],:8] = cswr_sw.iloc[:,1:9]
    data_matrix[count:count+cswr_sw.shape[0],8:16] = cswr_r.iloc[:,1:9]
    data_matrix[count:count+cswr_sw.shape[0],16] = np.ones((cswr_sw.shape[0],))*3

    return data_matrix

data_matrix_veh = create_matrix_features(data_pd,treat = 0)
data_matrix_cbd = create_matrix_features(data_pd,treat = 1)


### TRAIN GLOBAL CLASYFIEER
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
clf_1 = svm.SVC()

### pyramidal layer
pyr_feature_sel = [9,10,11,12,13,15,16]

pyramidal_ripples_index = np.where(data_matrix_veh[:,16] > 0)
pyramidal_ripples_matrix = data_matrix_veh[:,pyr_feature_sel ]
pyramidal_ripples_matrix = pyramidal_ripples_matrix[pyramidal_ripples_index,:]

pyramidal_ripples_index_cbd =  np.where(data_matrix_cbd[:,16] > 0)
pyramidal_ripples_matrix_cbd = data_matrix_cbd[:,pyr_feature_sel]
pyramidal_ripples_matrix_cbd = pyramidal_ripples_matrix_cbd[pyramidal_ripples_index_cbd,:]

scores_1_ripple = cross_val_score(clf_1, pyramidal_ripples_matrix[0,:,:-1], pyramidal_ripples_matrix[0,:,-1],cv = 10)
index = np.random.permutation(pyramidal_ripples_matrix.shape[1])
random_class = pyramidal_ripples_matrix[0,index,-1]
scores_1_ripple_random = cross_val_score(clf_1, pyramidal_ripples_matrix[0,:,:-1], random_class,cv =10)
clf_1.fit(pyramidal_ripples_matrix[0,:,:-1], pyramidal_ripples_matrix[0,:,-1])
score_1_pyramidal_veh_cbd = clf_1.score(pyramidal_ripples_matrix_cbd[0,:,:-1], pyramidal_ripples_matrix_cbd[0,:,-1])

scores_1_ripple_cbd = cross_val_score(clf_1, pyramidal_ripples_matrix_cbd[0,:,:-1], pyramidal_ripples_matrix_cbd[0,:,-1],cv = 10)
index = np.random.permutation(pyramidal_ripples_matrix_cbd.shape[1])
random_class = pyramidal_ripples_matrix_cbd[0,index,-1]
scores_1_ripple_random_cbd = cross_val_score(clf_1, pyramidal_ripples_matrix_cbd[0,:,:-1], random_class,cv =10)


# import some data to play with
class_names = ['R','SWR','cSWR']
X = pyramidal_ripples_matrix[0,:,:-1]
y = pyramidal_ripples_matrix[0,:,-1]

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC().fit(X_train, y_train)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Blues,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.savefig(figure_path + 'confusion_pyr.png')
plt.show()

### SW layer

sw_feature_sel = [1,2,3,4,5,7,16]
sw_index = np.where(data_matrix_veh[:,16] != 1)
sw_matrix = data_matrix_veh[:,sw_feature_sel]
sw_matrix = sw_matrix[sw_index,:]

sw_index_cbd =  np.where(data_matrix_cbd[:,16] != 1)
sw_matrix_cbd = data_matrix_cbd[:,sw_feature_sel]
sw_matrix_cbd = sw_matrix_cbd[sw_index_cbd,:]

scores_1_sw = cross_val_score(clf_1, sw_matrix[0,:,:-1], sw_matrix[0,:,-1],cv = 10)
index = np.random.permutation(sw_matrix.shape[1])
random_class = sw_matrix[0,index,-1]
scores_1_sw_random = cross_val_score(clf_1, sw_matrix[0,:,:-1], random_class,cv =10)
clf_1.fit(sw_matrix[0,:,:-1], sw_matrix[0,:,-1])
score_1_sw_veh_cbd= clf_1.score(sw_matrix_cbd[0,:,:-1], sw_matrix_cbd[0,:,-1])

scores_1_sw_cbd = cross_val_score(clf_1, sw_matrix_cbd[0,:,:-1], sw_matrix_cbd[0,:,-1],cv = 10)
index = np.random.permutation(sw_matrix_cbd.shape[1])
random_class = sw_matrix_cbd[0,index,-1]
scores_1_sw_random_cbd = cross_val_score(clf_1, sw_matrix_cbd[0,:,:-1], random_class,cv =10)

# import some data to play with
class_names = ['SW','SWR','cSWR']
X = sw_matrix[0,:,:-1]
y = sw_matrix[0,:,-1]
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC().fit(X_train, y_train)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Reds,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.savefig(figure_path + 'confusion_sw.png')
plt.show()

### complete information

complete_index_sel = [1,2,3,4,5,7,9,10,11,12,13,15]
scores_1_complete = cross_val_score(clf_1, data_matrix_veh[:,complete_index_sel], data_matrix_veh[:,16],cv = 10)
index = np.random.permutation(data_matrix_veh.shape[0])
random_class = data_matrix_veh[index,16]
scores_1_complete_veh_random = cross_val_score(clf_1, data_matrix_veh[:,complete_index_sel], random_class,cv =10)
clf_1.fit(data_matrix_veh[:,complete_index_sel],data_matrix_veh[:,16])
score1_complete_veh_cbd= clf_1.score(data_matrix_cbd[:,complete_index_sel],data_matrix_cbd[:,16])

scores_1_complete_cbd = cross_val_score(clf_1, data_matrix_cbd[:,complete_index_sel], data_matrix_cbd[:,16],cv = 10)
index = np.random.permutation(data_matrix_cbd.shape[0])
random_class = data_matrix_cbd[index,16]
scores_1_complete_random_cbd = cross_val_score(clf_1, data_matrix_cbd[:,complete_index_sel], random_class,cv =10)

# import some data to play with
class_names = ['SW','R','SWR','cSWR']
X = data_matrix_veh[:,complete_index_sel]
y =  data_matrix_veh[:,16]
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# Run classifier, using a model that is too regularized (C too low) to see
# the impact on the results
classifier = svm.SVC().fit(X_train, y_train)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
titles_options = [
    ("Confusion matrix, without normalization", None),
    ("Normalized confusion matrix", "true"),
]
for title, normalize in titles_options:
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels=class_names,
        cmap=plt.cm.Purples,
        normalize=normalize,
    )
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.savefig(figure_path + 'confusion_pyr_sw.png')
plt.show()

#### Now plotting

figure, axes = plt.subplots()

axes.scatter(np.ones(10),scores_1_ripple,c = 'k',alpha = 0.5)
axes.scatter(np.ones(10),scores_1_ripple_random,c = 'b',alpha = 0.5)
axes.scatter(2*np.ones(10),scores_1_ripple_cbd,c = 'g',alpha = 0.5)
axes.scatter(2*np.ones(10),scores_1_ripple_random_cbd,c = 'b',alpha = 0.5)
axes.scatter([3],score_1_pyramidal_veh_cbd,c = 'g',alpha = 0.5)

axes.scatter(5*np.ones(10),scores_1_sw,c = 'k',alpha = 0.5)
axes.scatter(5*np.ones(10),scores_1_sw_random,c = 'r',alpha = 0.5)
axes.scatter(6*np.ones(10),scores_1_sw_cbd,c = 'g',alpha = 0.5)
axes.scatter(6*np.ones(10),scores_1_sw_random_cbd,c = 'r',alpha = 0.5)
axes.scatter([7],score_1_sw_veh_cbd,c = 'g',alpha = 0.5)

axes.scatter(9*np.ones(10),scores_1_complete,c = 'k',alpha = 0.5)
axes.scatter(9*np.ones(10),scores_1_complete_veh_random,c = 'violet',alpha = 0.5)
axes.scatter(10*np.ones(10),scores_1_complete_cbd,c = 'g',alpha = 0.5)
axes.scatter(10*np.ones(10),scores_1_complete_random_cbd,c = 'violet',alpha = 0.5)
axes.scatter([11],score1_complete_veh_cbd,c = 'g',alpha = 0.5)


xlabel = ['','PYR_VEH_VEH','PYR_CBD_CBD','PYR_VEH_CBD','','SW_VEH_VEH','SW_CBD_CBD','SW_VEH_CBD','','ALL_VEH_VEH','ALL_CBD_CBD','ALL_VEH_CBD']

axes.set_xticks(np.arange(0,len(xlabel)))
axes.set_xticklabels(axes.get_xticks(), rotation = 45)
axes.set_xticklabels(xlabel)
axes.set_xlabel('CONDTIONS TRAINING vs TESTING', fontsize = 20)
axes.set_ylabel('Score', fontsize = 20)
axes.hlines(0.33,0,12,color = 'k',alpha = 0.5)
axes.hlines(0.25,0,12,color = 'k',alpha = 0.5)

axes.set_ylim([0,1])
axes.set_title('SVM classifier',fontsize = 30)
figure.set_size_inches([12,10])
figure.savefig(figure_path + 'SVM_R_SW_classifier_reduced_features.png')

plt.show()

#######################################

data_veh = data_matrix_veh[:,complete_index_sel]
data_cbd = data_matrix_cbd[:,complete_index_sel]
pca_veh = PCA()
pca_veh.fit(data_veh)
transformed_data_veh = pca_veh.transform(data_veh )
pca_cbd = PCA()
pca_cbd.fit(data_cbd)
transformed_data_cbd = pca_cbd.transform(data_cbd)


figure = plt.figure()
gs = plt.GridSpec(8,8)

axes0 = figure.add_subplot(gs[0:3, 0:3])  # , projection='3d')
axes0.scatter(np.arange(0,data_veh.shape[1]),pca_veh.explained_variance_ratio_,c= 'k',s = 5, marker = 'o')
axes0.scatter(np.arange(0,data_cbd.shape[1]),pca_cbd.explained_variance_ratio_,c= 'g', s = 5 , marker = 'x')
axes0.set_ylim([0,1])
axes0.set_xlabel('Rank',fontsize = 15)
axes0.set_ylabel('Explained Variance Ratio')
axes0.legend(['VEH','CBD'],fontsize = 15)
axes0.set_title('Eigenspectrum', fontsize = 20)

xticklabel = ['MF_SW','Amp_SW','AUC_SW','P2P_SW','Ent_SW','MF_R','Amp_R','AUC_R','P2P_R','Ent_R']
axes1 = figure.add_subplot(gs[0:3, 4:9])  # , projection='3d')
axes1.plot(np.arange(0,data_veh.shape[1]),pca_veh.components_[0,:], c = 'k')
axes1.plot(np.arange(0,data_veh.shape[1]),pca_cbd.components_[0,:], c = 'g')
axes1.set_xticks(np.arange(0,len(xticklabel)))
axes1.set_xticklabels(axes1.get_xticks(), rotation = 45)
axes1.set_xticklabels(xticklabel)
axes1.set_ylim([0,1])
axes1.set_ylabel('Score',fontsize = 15)
axes1.set_xlabel('Feature', fontsize = 15)
axes1.legend(['VEH','CBD'],fontsize = 15)
axes1.set_title('First Component', fontsize = 20)

color_list = ['red', 'blue', 'purple', 'yellow']
axes2 = figure.add_subplot(gs[5:9, 0:3])#, projection='3d')
for i in range(4):
    index = np.where(data_matrix_veh[:,16] == i)
    axes2.scatter(transformed_data_veh[index, 0], transformed_data_veh[index, 1], marker='o', color=color_list[i],
                  alpha=0.1)
axes2.set_ylim([-1000,1000])
axes2.set_xlim([-1000,4000])
axes2.set_xlabel('PC1',fontsize = 15)
axes2.set_ylabel('PC2',fontsize = 15)
axes2.set_title('VEHICLE', fontsize = 20)

axes3 = figure.add_subplot(gs[5:9, 5:9])#, projection='3d')
for i in range(4):
    index = np.where(data_matrix_cbd[:,16] == i)
    axes3.scatter(transformed_data_cbd[index, 0], transformed_data_cbd[index, 1], marker='x', color=color_list[i],
                  alpha=0.1)
axes3.set_ylim([-1000,1000])
axes3.set_xlim([-1000,4000])
axes3.set_xlabel('PC1',fontsize = 15)
axes3.set_ylabel('PC2',fontsize = 15)
axes3.set_title('CBD', fontsize = 20)

figure.set_size_inches([10,10])
figure.savefig(figure_path + 'pca_all_features.png')
plt.show()



figure = plt.figure()
gs = plt.GridSpec(8,8)

axes0 = figure.add_subplot(gs[0:3, 0:3])  # , projection='3d')
axes0.scatter(np.arange(0,data_veh.shape[1]),pca_veh.explained_variance_ratio_,c= 'k',s = 5, marker = 'o')
axes0.scatter(np.arange(0,data_cbd.shape[1]),pca_cbd.explained_variance_ratio_,c= 'g', s = 5 , marker = 'x')
axes0.set_ylim([0,1])
axes0.set_xlabel('Rank',fontsize = 15)
axes0.set_ylabel('Explained Variance Ratio')
axes0.legend(['VEH','CBD'],fontsize = 15)
axes0.set_title('Eigenspectrum', fontsize = 20)

xticklabel = ['MF_SW','Amp_SW','AUC_SW','P2P_SW','Ent_SW','MF_R','Amp_R','AUC_R','P2P_R','Ent_R']
axes1 = figure.add_subplot(gs[0:3, 4:9])  # , projection='3d')
axes1.plot(np.arange(0,data_veh.shape[1]),pca_veh.components_[0,:], c = 'k')
axes1.plot(np.arange(0,data_veh.shape[1]),pca_cbd.components_[0,:], c = 'g')
axes1.set_xticks(np.arange(0,len(xticklabel)))
axes1.set_xticklabels(axes1.get_xticks(), rotation = 45)
axes1.set_xticklabels(xticklabel)
axes1.set_ylim([0,1])
axes1.set_ylabel('Score',fontsize = 15)
axes1.set_xlabel('Feature', fontsize = 15)
axes1.legend(['VEH','CBD'],fontsize = 15)
axes1.set_title('First Component', fontsize = 20)

color_list = ['red', 'blue', 'purple', 'yellow']
axes2 = figure.add_subplot(gs[5:9, 0:3], projection='3d')
for i in range(4):
    index = np.where(data_matrix_veh[:,16] == i)
    axes2.scatter(transformed_data_veh[index, 0], transformed_data_veh[index, 1],transformed_data_veh[index, 2],marker='o', color=color_list[i],
                  alpha=0.1)
axes2.set_ylim([-1000,1000])
axes2.set_xlim([-500,4000])
axes2.set_xlabel('PC1',fontsize = 15)
axes2.set_ylabel('PC2',fontsize = 15)
axes2.set_zlabel('PC3',fontsize = 15)
axes2.set_title('VEHICLE', fontsize = 20)
axes2.set_zlim([-500,1000])

axes3 = figure.add_subplot(gs[5:9, 5:9], projection='3d')
for i in range(4):
    index = np.where(data_matrix_cbd[:,16] == i)
    axes3.scatter(transformed_data_cbd[index, 0], transformed_data_cbd[index, 1],transformed_data_cbd[index, 2],marker='x', color=color_list[i],
                  alpha=0.1)
axes3.set_ylim([-1000,1000])
axes3.set_xlim([-500,4000])
axes3.set_zlim([-500,1000])

axes3.set_xlabel('PC1',fontsize = 15)
axes3.set_ylabel('PC2',fontsize = 15)
axes3.set_zlabel('PC3',fontsize = 15)

axes3.set_title('CBD', fontsize = 20)

figure.set_size_inches([10,10])
figure.savefig(figure_path + 'pca_all_features3D.png')
plt.show()

####################3

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf_veh = LinearDiscriminantAnalysis()
clf_veh.fit(data_veh , data_matrix_veh[:,16])
X_veh = clf_veh.transform(data_veh)

clf_cbd = LinearDiscriminantAnalysis()
clf_cbd.fit(data_cbd , data_matrix_cbd[:,16])
X_cbd = clf_cbd.transform(data_cbd)


figure = plt.figure()
gs = plt.GridSpec(1,2)

color_list = ['red', 'blue', 'purple', 'yellow']
axes2 = figure.add_subplot(gs[0, 0], projection='3d')
for i in range(4):
    index = np.where(data_matrix_veh[:,16] == i)
    axes2.scatter(X_veh[index, 0], X_veh[index, 1],X_veh[index, 2],marker='o', color=color_list[i],
                  alpha=0.1)
# axes2.set_ylim([-1000,1000])
# axes2.set_xlim([-500,4000])
# axes2.set_zlim([-500,1000])

axes2.set_xlabel('LD1',fontsize = 15)
axes2.set_ylabel('LD2',fontsize = 15)
axes2.set_zlabel('LD3',fontsize = 15)
axes2.set_title('VEHICLE', fontsize = 20)

axes3 = figure.add_subplot(gs[0, 1], projection='3d')
for i in range(4):
    index = np.where(data_matrix_cbd[:,16] == i)
    axes3.scatter(X_cbd[index, 0], X_cbd[index, 1],X_cbd[index, 2],marker='x', color=color_list[i],
                  alpha=0.1)
# axes3.set_ylim([-1000,1000])
# axes3.set_xlim([-500,4000])
# axes3.set_zlim([-500,1000])
axes3.set_xlabel('LD1',fontsize = 15)
axes3.set_ylabel('LD2',fontsize = 15)
axes3.set_zlabel('LD3',fontsize = 15)
axes3.set_title('CBD', fontsize = 20)

figure.set_size_inches([10,5])
figure.savefig(figure_path + 'lda_all_features.png')
plt.show()


###


figure = plt.figure()
gs = plt.GridSpec(1,2)

color_list = ['red', 'blue', 'purple', 'yellow']
axes2 = figure.add_subplot(gs[0, 0])
for i in range(4):
    index = np.where(data_matrix_veh[:,16] == i)
    axes2.scatter(X_veh[index, 0], X_veh[index, 1],marker='o', color=color_list[i],
                  alpha=0.1)
axes2.set_ylim([-10,10])
axes2.set_xlim([-5,10])
# axes2.set_zlim([-500,1000])

axes2.set_xlabel('LD1',fontsize = 15)
axes2.set_ylabel('LD2',fontsize = 15)
axes2.set_title('VEHICLE', fontsize = 20)

axes3 = figure.add_subplot(gs[0, 1])
for i in range(4):
    index = np.where(data_matrix_cbd[:,16] == i)
    axes3.scatter(X_cbd[index, 0], X_cbd[index, 1],marker='x', color=color_list[i],
                  alpha=0.1)
axes3.set_ylim([-10,10])
axes3.set_xlim([-5,10])
# axes3.set_zlim([-500,1000])
axes3.set_xlabel('LD1',fontsize = 15)
axes3.set_ylabel('LD2',fontsize = 15)
axes3.set_title('CBD', fontsize = 20)

figure.set_size_inches([10,5])
figure.savefig(figure_path + 'lda_all_features_2d_0.png')
plt.show()


