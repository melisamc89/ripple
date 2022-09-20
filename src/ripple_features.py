
import numpy as np
import scipy.io as sio
from scipy.signal import hilbert

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

def load_data_features_by_animal(data_path,ID_sequence,events_sequence,time2):

    events_counter = np.zeros((4, len(ID_sequence)))
    class_vector_list = []
    ripple_list = []
    for rat_number in range(len(ID_sequence)):
        data_file = 'HPCpyra_events_ratID' + str(ID_sequence[rat_number]) + '.mat'
        data = sio.loadmat(data_path + data_file)
        for key in range(3):
            ripple_data = data[events_sequence[key]]
            events_counter[key, rat_number] += ripple_data.shape[0]
            events_counter[3, rat_number] += ripple_data.shape[0]
        class_vector = np.zeros((int(events_counter[3,rat_number]),))
        counter = 0
        ripple_matrix = np.zeros((int(events_counter[3,rat_number]),len(time2)))
        for key in range(3):
            ripple_data = data[events_sequence[key]]
            ripple_matrix[int(counter):int(counter+ripple_data.shape[0])]= ripple_data[:,time[time2]]
            class_vector[int(counter):int(counter+ripple_data.shape[0])] = key
            counter+=int(ripple_data.shape[0])
        ripple_list.append(ripple_matrix)
        class_vector_list.append(class_vector)

    return ripple_list, class_vector_list

def load_bin_information(data_path,ID_sequence,events_sequence,total_count,desired_variable):

    ###Vehicle
    bin_vector = np.zeros((total_count,))
    counter = 0
    for rat_number in range(len(ID_sequence)):
        data_file = 'GC_ratID' + str(ID_sequence[rat_number]) + '_cbd.mat'
        data = sio.loadmat(data_path + data_file)
        data = data['data']
        for key in range(3):
            ripple_info_bin = data[events_sequence[key]][0][0][:, desired_variable]
            bin_vector[counter:counter+ripple_info_bin.shape[0]]=ripple_info_bin
            counter+=ripple_info_bin.shape[0]

    return bin_vector

def load_features(data_path,ID_sequence,events_sequence,total_count,group):

    features = np.zeros((total_count,8))
    counter = 0
    for rat_number in range(len(ID_sequence)):
        data_file = 'GC_ratID' + str(ID_sequence[rat_number]) +'_'+group +'.mat'
        data = sio.loadmat(data_path + data_file)
        data = data['data']
        for key in range(3):
            ripple_info_bin = data[events_sequence[key]][0][0]
            features[counter:counter+ripple_info_bin.shape[0],:]=ripple_info_bin
            counter+=ripple_info_bin.shape[0]

    return features

def load_data_features_by_animal(data_path,ID_sequence,events_sequence,time2):

    events_counter = np.zeros((4, len(ID_sequence)))
    class_vector_list = []
    ripple_list = []
    for rat_number in range(len(ID_sequence)):
        data_file = 'HPCpyra_events_ratID' + str(ID_sequence[rat_number]) + '.mat'
        data = sio.loadmat(data_path + data_file)
        for key in range(3):
            ripple_data = data[events_sequence[key]]
            events_counter[key, rat_number] += ripple_data.shape[0]
            events_counter[3, rat_number] += ripple_data.shape[0]
        class_vector = np.zeros((int(events_counter[3,rat_number]),))
        counter = 0
        ripple_matrix = np.zeros((int(events_counter[3,rat_number]),len(time2)))
        for key in range(3):
            ripple_data = data[events_sequence[key]]
            ripple_matrix[int(counter):int(counter+ripple_data.shape[0])]= ripple_data[:,time[time2]]
            class_vector[int(counter):int(counter+ripple_data.shape[0])] = key
            counter+=int(ripple_data.shape[0])
        ripple_list.append(ripple_matrix)
        class_vector_list.append(class_vector)

    return ripple_list, class_vector_list, events_counter

def load_features_by_animal(data_path,ID_sequence,events_sequence,events_counter,group):

    features_list = []
    for rat_number in range(len(ID_sequence)):
        features = np.zeros((int(events_counter[3,rat_number]), 8))
        data_file = 'GC_ratID' + str(ID_sequence[rat_number]) +'_'+group +'.mat'
        data = sio.loadmat(data_path + data_file)
        data = data['data']
        counter = 0
        for key in range(3):
            ripple_info_bin = data[events_sequence[key]][0][0]
            features[int(counter):int(counter+ripple_info_bin.shape[0]),:]=ripple_info_bin
            counter+=ripple_info_bin.shape[0]
        features_list.append(features)

    return features_list


rat_ID_veh = [3,4,9,201,203,206,210,211,213]
rat_ID_cbd= [2,5,10,11,204,205,207,209,212,214]

keywords= ['cwr','r','swr']
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

total_ripple_count_veh, total_matrix_veh= count_events(data_path_veh,rat_ID_veh,keywords_veh)
percetage_ripple_count_veh = total_ripple_count_veh / total_ripple_count_veh[3]
total_ripple_count_cbd, total_matrix_cbd = count_events(data_path_cbd,rat_ID_cbd,keywords_cbd)
percetage_ripple_count_cbd = total_ripple_count_cbd / total_ripple_count_cbd[3]
ripple_veh_matrix_list, class_vector_veh= load_all_data(data_path_veh,rat_ID_veh,keywords_veh,total_ripple_count_veh,time2)
ripple_cbd_matrix_list, class_vector_cbd = load_all_data(data_path_cbd,rat_ID_cbd,keywords_cbd,total_ripple_count_cbd,time2)

data_path= '/home/melisamc/Documentos/ripple/data/features_output/'

features_veh = load_features(data_path, rat_ID_veh, keywords, int(total_ripple_count_veh[3]),'veh')
features_cbd = load_features(data_path, rat_ID_cbd, keywords, int(total_ripple_count_cbd[3]),'cbd')

features_label = ['MeanFreq','Amplitude','Area','Duration','Peak2Peak','Power','Entropy','NumberOfPeaks']
figure, axes = plt.subplots(2,4)
order_veh = np.arange(0,features_veh.shape[0])/features_veh.shape[0]
order_cbd = np.arange(0,features_cbd.shape[0])/features_cbd.shape[0]
for i in range(2):
    for j in range(4):
        index = i*4 + j
        sorted_veh = np.sort(features_veh[:,index])
        sorted_cbd = np.sort(features_cbd[:,index])
        axes[i,j].scatter(sorted_veh,order_veh, color = 'k')
        axes[i,j].scatter(sorted_cbd,order_cbd,color = 'g')
        axes[i,j].set_ylabel('Cumulative',fontsize = 20)
        axes[i,j].set_xlabel(features_label[index],fontsize = 20)
        # axes[i, j].set_xscale('log')
        # axes[i, j].set_yscale('log')
        axes[i,j].legend(['VEH','CBD'])
figure.set_size_inches([25,10])
figure.suptitle('Cumulative Distributions by order', fontsize = 25)
figure.savefig(figure_path + 'features_2.png')
plt.show()

color_list = ['b','r','g']
color_list_2 = ['cyan','magenta','yellow']


figure, axes = plt.subplots(2,4)
for i in range(2):
    for j in range(4):
        index = i*4 + j
        for key in range(3):
            index2_veh = np.where(class_vector_veh == key)[0]
            index2_cbd = np.where(class_vector_cbd == key)[0]
            sorted_veh = np.sort(features_veh[index2_veh,index])
            sorted_cbd = np.sort(features_cbd[index2_cbd,index])
            order_veh = np.arange(0, index2_veh.shape[0])/index2_veh.shape[0]
            order_cbd = np.arange(0, index2_cbd.shape[0])/index2_cbd.shape[0]
            axes[i,j].scatter(sorted_veh,order_veh, color = color_list[key],alpha = 0.5)
            axes[i,j].scatter(sorted_cbd,order_cbd,color = color_list_2[key])
            axes[i,j].set_ylabel('Cumulative',fontsize = 20)
            axes[i,j].set_xlabel(features_label[index],fontsize = 20)
            axes[i,j].set_xscale('log')
            axes[i,j].set_yscale('log')
axes[1,3].legend(['CSWR_VEH','CSWR_CBD','R_VEH','R_CBD','SWR_VEH','SWR_CBD'])
figure.set_size_inches([25,10])
figure.suptitle('Order features by Ripple Type', fontsize = 25)
figure.savefig(figure_path + 'features_class_2_log.png')
plt.show()

# pca = PCA()
# pca.fit(features_veh)
# x = pca.transform(features_veh)

### TRAIN GLOBAL CLASYFIEER
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

clf_1 = svm.SVC()
scores_1_veh = cross_val_score(clf_1, features_veh, class_vector_veh,cv = 10)
clf_1.fit(features_veh,class_vector_veh)
score1_veh_cbd= clf_1.score(features_cbd,class_vector_cbd)
index = np.random.permutation(len(class_vector_veh))
random_class = class_vector_veh[index]
scores_1_veh_random = cross_val_score(clf_1, features_veh, random_class,cv =10)

clf_2 = GaussianNB()
scores_2_veh = cross_val_score(clf_2, features_veh, class_vector_veh,cv =10)
clf_2.fit(features_veh,class_vector_veh)
score2_veh_cbd = clf_2.score(features_cbd,class_vector_cbd)
scores_2_veh_random = cross_val_score(clf_2, features_veh, random_class,cv =10)

clf_1 = svm.SVC()
scores_1_cbd = cross_val_score(clf_1, features_cbd, class_vector_cbd,cv=10)
index = np.random.permutation(len(class_vector_cbd))
random_class = class_vector_cbd[index]
scores_1_cbd_random = cross_val_score(clf_1, features_cbd, random_class,cv=10)

clf_2 = GaussianNB()
scores_2_cbd = cross_val_score(clf_2, features_cbd, class_vector_cbd,cv=10)
scores_2_cbd_random = cross_val_score(clf_2, features_cbd, random_class)


vector1 = [1,1,1,1,1,1,1,1,1,1]
vector2 = [2]

vector3 = [4,4,4,4,4,4,4,4,4,4]
vector4 = [5]

xlabel = ['','VEH_VEH','VEH_CBD','','CBD_CBD','']

# figure, axes = plt.subplots()
# axes.scatter(vector1,scores_1_veh,color = 'b',alpha= 0.2)
# axes.scatter(vector1,scores_1_veh_random,color = 'k', alpha = 0.2)
# axes.errorbar([1],np.mean(scores_1_veh),np.std(scores_1_veh),color = 'b')
# axes.errorbar([1],np.mean(scores_1_veh_random),np.std(scores_1_veh_random),color ='k')
# axes.scatter(vector2,score1_veh_cbd, color ='r',alpha = 0.5)
#
# axes.scatter(vector3,scores_1_cbd,color = 'b',alpha= 0.2)
# axes.scatter(vector3,scores_1_cbd_random,color = 'k', alpha = 0.2)
# axes.errorbar([1],np.mean(scores_1_cbd),np.std(scores_1_cbd),color = 'b',markdown='x')
# axes.errorbar([1],np.mean(scores_1_cbd_random),np.std(scores_1_cbd_random),color ='k')
# axes.set_xticks(np.arange(0,6))
# axes.set_xticklabels(xlabel)
# axes.set_xlabel('CONDTIONS TRAINING vs TESTING', fontsize = 15)
# axes.set_ylabel('Score', fontsize = 15)
#
# axes.hlines(0.33,0,5,color = 'k')
# axes.set_ylim([0,1])
# axes.set_title('SVM classifier',fontsize = 20)
# figure.savefig(figure_path + 'SVM_classifier.png')
#
# plt.show()
#
#
# figure, axes = plt.subplots()
# axes.scatter(vector1,scores_2_veh,color = 'b',alpha= 0.5)
# axes.scatter(vector1,scores_2_veh_random,color = 'k', alpha = 0.5)
# axes.scatter(vector2,score2_veh_cbd, color ='r',alpha = 0.5)
# axes.errorbar(vector1,np.mean(scores_2_veh),np.std(scores_2_veh))
# axes.scatter(vector2,score2_veh_cbd, color ='r',alpha = 0.5)
#
# axes.scatter(vector3,scores_2_cbd,color = 'b',alpha= 0.5)
# axes.scatter(vector3,scores_2_cbd_random,color = 'k', alpha = 0.5)
# axes.set_xticks(np.arange(0,6))
# axes.set_xticklabels(xlabel)
# axes.set_xlabel('CONDTIONS TRAINING vs TESTING', fontsize = 15)
# axes.set_ylabel('Score', fontsize = 15)
#
# axes.hlines(0.33,0,5,color = 'k')
# axes.set_ylim([0,1])
# axes.set_title('GNB classifier',fontsize = 20)
# figure.savefig(figure_path + 'GNB_classifier.png')
#
# plt.show()

###############################################################################

ripple_veh_list, class_veh_list, events_count_veh = load_data_features_by_animal(data_path_veh,rat_ID_veh,keywords_veh,time2)
ripple_cbd_list, class_cbd_list, events_count_cbd = load_data_features_by_animal(data_path_cbd,rat_ID_cbd,keywords_cbd,time2)

features_veh_list = load_features_by_animal(data_path,rat_ID_veh,keywords,events_count_veh,'veh')
features_cbd_list = load_features_by_animal(data_path,rat_ID_cbd,keywords,events_count_cbd,'cbd')

features_list = []
class_list = []
for i in range(len(features_veh_list)):
    features_list.append(features_veh_list[i])
    class_list.append(class_veh_list[i])
for i in range(len(features_cbd_list)):
    features_list.append(features_cbd_list[i])
    class_list.append(class_cbd_list[i])

scores_veh = np.zeros((len(rat_ID_veh),))
clf = GaussianNB()
clf = svm.SVC()
for rat in range(len(rat_ID_veh)):
    index = np.random.permutation(features_veh_list[rat].shape[0])
    X_train, X_test, y_train, y_test = train_test_split(features_veh_list[rat][index,:], class_veh_list[rat][index], test_size = 0.1, random_state = 0)
    clf.fit(X_train, y_train)
    scores_veh[rat] = clf.score(X_test, y_test)

scores_cbd = np.zeros((len(rat_ID_cbd,)))
for rat in range(len(rat_ID_cbd)):
    index = np.random.permutation(features_cbd_list[rat].shape[0])
    X_train, X_test, y_train, y_test = train_test_split(features_cbd_list[rat][index,:], class_cbd_list[rat][index], test_size = 0.1, random_state = 0)
    clf.fit(X_train, y_train)
    scores_cbd[rat] = clf.score(X_test, y_test)

scores_matrix_cross_val = np.zeros((10,len(features_list),len(features_list)))
for cross in range(10):
    training_list_X = []
    training_list_y = []
    testing_list_X = []
    testing_list_y = []
    for rat1 in range(len(features_list)):
        index = np.random.permutation(features_list[rat].shape[0])
        X_train, X_test, y_train, y_test = train_test_split(features_list[rat][index, :], class_list[rat][index],
                                                            test_size=0.1, random_state=0)
        training_list_X.append(X_train)
        training_list_y.append(y_train)
        testing_list_X.append(X_test)
        testing_list_y.append(y_test)

    scores_matrix = np.zeros((len(features_list),len(features_list)))
    for rat1 in range(len(features_list)):
        clf.fit(training_list_X[rat1], training_list_y[rat1])
        for rat2 in range(len(features_list)):
            scores_matrix[rat1,rat2] = clf.score(testing_list_X[rat2], testing_list_y[rat2])
    scores_matrix_cross_val[cross,:,:] = scores_matrix


scores_matrix_cross_val_random = np.zeros((10,len(features_list),len(features_list)))
for cross in range(10):
    training_list_X = []
    training_list_y = []
    testing_list_X = []
    testing_list_y = []
    for rat1 in range(len(features_list)):
        index = np.random.permutation(features_list[rat].shape[0])
        X_train, X_test, y_train, y_test = train_test_split(features_list[rat], class_list[rat][index],
                                                            test_size=0.1, random_state=0)
        training_list_X.append(X_train)
        training_list_y.append(y_train)
        testing_list_X.append(X_test)
        testing_list_y.append(y_test)

    scores_matrix = np.zeros((len(features_list),len(features_list)))
    for rat1 in range(len(features_list)):
        clf.fit(training_list_X[rat1], training_list_y[rat1])
        for rat2 in range(len(features_list)):
            scores_matrix[rat1,rat2] = clf.score(testing_list_X[rat2], testing_list_y[rat2])
    scores_matrix_cross_val_random[cross,:,:] = scores_matrix

import matplotlib.patches as patches

dx, dy = 1, 1
y, x = np.mgrid[slice(0, 19 + dy, dy),
                slice(0, 19 + dx, dx)]

rats_label = []
for i in range(len(rat_ID_veh)):
    rats_label.append('VEH_'+str(rat_ID_veh[i]))
for i in range(len(rat_ID_cbd)):
    rats_label.append('CBD_'+str(rat_ID_cbd[i]))

figure, axes = plt.subplots(2,3)
# reversing the original colormap using reversed() function
orig_map=plt.cm.get_cmap('gray')
reversed_map = orig_map.reversed()
ax0 = axes[0,0]
pcm0 = ax0.pcolormesh(x[0:19,0:19],y[0:19,0:19],np.mean(scores_matrix_cross_val,axis=0)[0:19,0:19],vmin = 0.5,vmax = 0.85,cmap=reversed_map )
figure.colorbar(pcm0, ax=ax0)
#pcm0 = ax0.pcolormesh(x[0:9,0:9],y[0:9,0:9],np.mean(scores_matrix_cross_val,axis=0)[0:9,0:9],vmin = 0.55,vmax = 0.78,cmap='Blues')
#pcm0 = ax0.pcolormesh(x[9:19,9:19],y[9:19,9:19],np.mean(scores_matrix_cross_val,axis=0)[9:19,9:19],vmin = 0.55,vmax = 0.78,cmap='Greens')
# Create a Rectangle patch
rect = patches.Rectangle((8.5,8.5), 10, 10, linewidth=3, edgecolor='g', facecolor='none')
# Add the patch to the Axes
ax0.add_patch(rect)
rect = patches.Rectangle((-0.5, -0.5),9, 9, linewidth=3, edgecolor='b', facecolor='none')
ax0.add_patch(rect)
ax0.set_xticks(np.arange(0,19))
ax0.set_xticklabels(ax0.get_xticks(), rotation = 45)
ax0.set_xticklabels(rats_label)
ax0.set_yticks(np.arange(0,19))
ax0.set_yticklabels(rats_label)
ax0.set_title('Cross Rat Accuracy',fontsize = 20)
ax0.legend(['VEH','CBD'])
ax0.set_xlabel('Trainning Set',fontsize = 15)
ax0.set_ylabel('Testing Set',fontsize = 15)

ax1 = axes[0,1]
pcm1 = ax1.pcolormesh(x[0:19,0:19],y[0:19,0:19],np.mean(scores_matrix_cross_val_random,axis=0)[0:19,0:19],vmin = 0.5,vmax = 0.85,cmap=reversed_map )
figure.colorbar(pcm1, ax=ax1)
# Create a Rectangle patch
rect = patches.Rectangle((8.5,8.5), 10, 10, linewidth=3, edgecolor='g', facecolor='none')
# Add the patch to the Axes
ax1.add_patch(rect)
rect = patches.Rectangle((-0.5, -0.5),9, 9, linewidth=3, edgecolor='b', facecolor='none')
ax1.add_patch(rect)
ax1.set_xticks(np.arange(0,19))
ax1.set_xticklabels(ax0.get_xticks(), rotation = 45)
ax1.set_xticklabels(rats_label)
ax1.set_yticks(np.arange(0,19))
ax1.set_yticklabels(rats_label)
ax1.set_title('Cross Rat Accuracy RANDOM',fontsize = 20)
ax1.set_xlabel('Trainning Set',fontsize = 15)
ax1.set_ylabel('Testing Set',fontsize = 15)


ax2 = axes[0,2]
z_scored = (np.mean(scores_matrix_cross_val,axis=0)[0:19,0:19]-np.mean(scores_matrix_cross_val_random,axis=0)[0:19,0:19])/np.std(scores_matrix_cross_val_random,axis=0)[0:19,0:19]
pcm2 = ax2.pcolormesh(x[0:19,0:19],y[0:19,0:19],z_scored,vmin = 1.6,vmax = 7,cmap = reversed_map)
figure.colorbar(pcm2, ax=ax2)
# Create a Rectangle patch
rect = patches.Rectangle((8.5,8.5), 10, 10, linewidth=3, edgecolor='g', facecolor='none')
# Add the patch to the Axes
ax2.add_patch(rect)
rect = patches.Rectangle((-0.5, -0.5),9, 9, linewidth=3, edgecolor='b', facecolor='none')
ax2.add_patch(rect)
ax2.set_xticks(np.arange(0,19))
ax2.set_xticklabels(ax0.get_xticks(), rotation = 45)
ax2.set_xticklabels(rats_label)
ax2.set_yticks(np.arange(0,19))
ax2.set_yticklabels(rats_label)
ax2.set_title('Z-scored accuracy',fontsize = 20)
ax2.set_xlabel('Trainning Set',fontsize = 15)
ax2.set_ylabel('Testing Set',fontsize = 15)

bins = [0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9]

ax3 = axes[1,0]
veh_accuracy = np.mean(scores_matrix_cross_val,axis=0)[0:9,0:9]
cbd_accuracy = np.mean(scores_matrix_cross_val,axis=0)[9:19,9:19]
veh_accuracy_random = np.mean(scores_matrix_cross_val_random,axis=0)[0:9,0:9]
cbd_accuracy_random = np.mean(scores_matrix_cross_val_random,axis=0)[9:19,9:19]
ax3.hist(veh_accuracy.flatten(),bins =bins, color = 'b',alpha = 0.5,density = True)
ax3.hist(cbd_accuracy.flatten(),bins = bins, color = 'g', alpha = 0.5,density = True)
ax3.hist(veh_accuracy_random.flatten(),bins =bins, color = 'b',alpha = 0.2,density = True)
ax3.hist(cbd_accuracy_random.flatten(),bins = bins, color = 'g', alpha = 0.2,density = True)
ax3.legend(['VEH','CBD','VEH_random','CBD_random'],fontsize = 15)
ax3.set_xlabel('Accuracy', fontsize = 15)
ax3.set_ylabel('#', fontsize = 15)
ax3.set_title('With In Treatment', fontsize = 20)

ax4 = axes[1,1]
veh_accuracy = np.mean(scores_matrix_cross_val,axis=0)[0:9,9:19]
cbd_accuracy = np.mean(scores_matrix_cross_val,axis=0)[9:19,0:9]
veh_accuracy_random = np.mean(scores_matrix_cross_val_random,axis=0)[0:9,9:19]
cbd_accuracy_random = np.mean(scores_matrix_cross_val_random,axis=0)[9:19,0:9]
ax4.hist(veh_accuracy.flatten(),bins =bins, color = 'b',alpha = 0.5,density = True)
ax4.hist(cbd_accuracy.flatten(),bins = bins, color = 'g', alpha = 0.5,density = True)
ax4.hist(veh_accuracy_random.flatten(),bins =bins, color = 'b',alpha = 0.2,density = True)
ax4.hist(cbd_accuracy_random.flatten(),bins = bins, color = 'g', alpha = 0.2,density = True)
ax4.legend(['VEH_CBD','CBD_VEH','VEH_CBD_random','CBD_VEH_random'],fontsize = 15)
ax4.set_xlabel('Accuracy', fontsize = 15)
ax4.set_ylabel('#', fontsize = 15)
ax4.set_title('Cross Treatment', fontsize = 20)


bins = np.arange(0,8)
ax4 = axes[1,2]
ax4.hist(z_scored[0:9,0:9].flatten(),bins =bins, color = 'b',alpha = 0.5,density = True)
ax4.hist(z_scored[9:19,9:19].flatten(),bins = bins, color = 'g', alpha = 0.5,density = True)
ax4.hist(z_scored[0:9,9:19].flatten(),bins =bins, color = 'b',alpha = 0.2,density = True)
ax4.hist(z_scored[9:19,0:9].flatten(),bins = bins, color = 'g', alpha = 0.2,density = True)
ax4.legend(['VEH_VEH','CBD_CBD','VEH_CBD','CBD_VEH'],fontsize = 15)
ax4.set_xlabel('Accuracy', fontsize = 15)
ax4.set_ylabel('#', fontsize = 15)
ax4.set_title('Z_scored Accuracy', fontsize = 20)

figure.set_size_inches([25,17])
figure.savefig(figure_path + 'SVM_class.png')
plt.show()
