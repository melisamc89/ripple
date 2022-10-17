import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


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

def load_bin_information(data_path,ID_sequence,events_sequence,total_count,group,desired_variable):

    ###Vehicle
    bin_vector = np.zeros((total_count,))
    counter = 0
    for rat_number in range(len(ID_sequence)):
        data_file = 'GC_ratID' + str(ID_sequence[rat_number]) + '_'+group+'.mat'
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

keywords_veh = ['HPCpyra_complex_swr_veh','HPCpyra_ripple_veh','HPCpyra_swr_veh']
keywords_cbd = ['HPCpyra_complex_swr_cbd','HPCpyra_ripple_cbd','HPCpyra_swr_cbd']
type_label = ['ComplexRipple','Ripple','SWR']
srate = 600

data_path_veh = '/home/melisamc/Documentos/ripple/data/HPCpyra/'
data_path_cbd = '/home/melisamc/Documentos/ripple/data/CBD/HPCpyra/'
data_path_group_info ='/home/melisamc/Documentos/ripple/data/group_info/'
data_path= '/home/melisamc/Documentos/ripple/data/features_output/'
figure_path = '/home/melisamc/Documentos/ripple/figures/'


time = np.arange(0, 3600)
time2 = np.arange(0, 3600)

total_ripple_count_veh, total_matrix_veh= count_events(data_path_veh,rat_ID_veh,keywords_veh)
percetage_ripple_count_veh = total_ripple_count_veh / total_ripple_count_veh[3]
total_ripple_count_cbd, total_matrix_cbd = count_events(data_path_cbd,rat_ID_cbd,keywords_cbd)
percetage_ripple_count_cbd = total_ripple_count_cbd / total_ripple_count_cbd[3]
ripple_veh_matrix_list, class_vector_veh= load_all_data(data_path_veh,rat_ID_veh,keywords_veh,total_ripple_count_veh,time2)
ripple_cbd_matrix_list, class_vector_cbd = load_all_data(data_path_cbd,rat_ID_cbd,keywords_cbd,total_ripple_count_cbd,time2)

keywords= ['cwr','r','swr']
features_veh = load_features(data_path, rat_ID_veh, keywords, int(total_ripple_count_veh[3]),'veh')
features_cbd = load_features(data_path, rat_ID_cbd, keywords, int(total_ripple_count_cbd[3]),'cbd')

keywords= ['cr','r','swr']
bin_vector_veh = load_bin_information(data_path_group_info, rat_ID_veh, keywords, int(total_ripple_count_veh[3]),'veh',desired_variable = 6)
bin_vector_cbd = load_bin_information(data_path_group_info, rat_ID_cbd, keywords, int(total_ripple_count_cbd[3]),'cbd',desired_variable = 6)

# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(max_depth=5, random_state=0)
from sklearn.metrics import confusion_matrix
# from sklearn.tree import DecisionTreeClassifier
# clf = DecisionTreeClassifier(random_state=0)
# from sklearn.tree import DecisionTreeRegressor
# clf = DecisionTreeRegressor(random_state=0)
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(random_state=5, max_iter=300)

#clf = GaussianNB()
clf = svm.SVC()

count_ripples = 0
for i in range(2,11):
    events = np.where(bin_vector_veh == i)[0]
    count_ripples+= events.shape[0]

features_veh_ = np.zeros((count_ripples,8))
bin_vector_veh_ = np.zeros((count_ripples,))
count_ripples = 0
for i in range(2,11):
    events = np.where(bin_vector_veh == i)[0]
    features_veh_[count_ripples:count_ripples+events.shape[0],:] = features_veh[events,:]
    bin_vector_veh_[count_ripples:count_ripples+events.shape[0]] = bin_vector_veh[events]
    count_ripples+= events.shape[0]

X_train, X_test, y_train, y_test = train_test_split(features_veh_, bin_vector_veh_,
                                                    test_size=0.1, random_state=0)

testing_units = 50
X_test_downsample = np.zeros((50*9,8))
y_test_downsample = np.zeros((50*9,))
count = 0
for i in range(2,11):
    bins_test = np.where(y_test == i)[0]
    bins_selected = bins_test[0:50]
    X_test_downsample[count:count+50,:] = X_test[bins_selected,:]
    y_test_downsample[count:count+50] = y_test[bins_selected]
    count += 50

clf.fit(X_train, y_train)
print(clf.score(X_test_downsample, y_test_downsample))
y_pred = clf.predict(X_test_downsample)
confusion_veh = confusion_matrix(y_test_downsample, y_pred)

count_ripples = 0
for i in range(2, 11):
    events = np.where(bin_vector_cbd == i)[0]
    count_ripples += events.shape[0]

features_cbd_ = np.zeros((count_ripples, 8))
bin_vector_cbd_ = np.zeros((count_ripples,))
count_ripples = 0
for i in range(2, 11):
    events = np.where(bin_vector_cbd == i)[0]
    features_cbd_[count_ripples:count_ripples + events.shape[0], :] = features_cbd[events, :]
    bin_vector_cbd_[count_ripples:count_ripples + events.shape[0]] = bin_vector_cbd[events]
    count_ripples+= events.shape[0]

X_train, X_test, y_train, y_test = train_test_split(features_cbd_, bin_vector_cbd_,
                                                    test_size=0.1, random_state=0)
testing_units = 50
X_test_downsample = np.zeros((50*9,8))
y_test_downsample = np.zeros((50*9,))
count = 0
for i in range(2,11):
    bins_test = np.where(y_test == i)[0]
    bins_selected = bins_test[0:50]
    X_test_downsample[count:count+50,:] = X_test[bins_selected,:]
    y_test_downsample[count:count+50] = y_test[bins_selected]
    count += 50
clf.fit(X_train, y_train)
print(clf.score(X_test_downsample, y_test_downsample))
y_pred = clf.predict(X_test_downsample)
confusion_cbd = confusion_matrix(y_test_downsample, y_pred)


dx, dy = 1, 1
y, x = np.mgrid[slice(2, 10 + dy, dy),
                slice(2, 10 + dx, dx)]

import matplotlib.patches as patches

figure, axes = plt.subplots(1,2)
orig_map=plt.cm.get_cmap('gray')
reversed_map = orig_map.reversed()
ax0 = axes[0]
pcm0 = ax0.pcolormesh(x,y,confusion_veh,cmap=reversed_map)#,vmin = 0, vmax = 8)
figure.colorbar(pcm0, ax=ax0)
axes[0].set_xlabel('Predicted BIN',fontsize = 15)
axes[0].set_ylabel('True BIN',fontsize = 15)
axes[0].set_title('VEHICLE',fontsize = 25)

orig_map=plt.cm.get_cmap('viridis')
reversed_map = orig_map.reversed()
ax1 = axes[1]
pcm1 = ax1.pcolormesh(x,y,confusion_cbd,cmap=reversed_map)#,vmin = 0, vmax = 8)
figure.colorbar(pcm1, ax=ax1)
rect = patches.Rectangle((5.5,4.5), 4, 5, linewidth=3, edgecolor='k', facecolor='none')
# Add the patch to the Axes
ax1.add_patch(rect)
axes[1].set_xlabel('Predicted BIN',fontsize = 15)
axes[1].set_ylabel('True BIN',fontsize = 15)
axes[1].set_title('CBD',fontsize = 25)

figure.set_size_inches([10,4])
figure.savefig(figure_path + 'cross_bin_confusion_matrix_SVM.png')
plt.show()
