
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import signal
from scipy.io import savemat


rat_ID = [3,4,9,201,203,211,213]
rat_test = [206,210]
keywords = ['HPCpyra_complex_swr_veh','HPCpyra_ripple_veh','HPCpyra_swr_veh']

rat_ID = [2,5,10,11,204,205,207,209,212,214]
rat_test = [204,205]
keywords = ['HPCpyra_complex_swr_cbd','HPCpyra_ripple_cbd','HPCpyra_swr_cbd']

type_label = ['ComplexRipple','Ripple','SWR']
srate = 600
rat_number = 0

data_path = '/home/melisamc/Documentos/ripple/data/CBD/HPCpyra/'
output_data_path = '/home/melisamc/Documentos/ripple/data/CBD_PCA_HPCpyra/'
figure_path = '/home/melisamc/Documentos/ripple/figures/cbd/'

color_list = ['b','r','g']
time = np.arange(1500, 2000)
time2 = np.arange(225, 375)

time = np.arange(0, 3600)
time2 = np.arange(0, 3600)

complex_ripple_count = 0
ripple_count = 0
swr_count = 0
count = 0
for rat_number in range(len(rat_ID)):
    data_file = 'HPCpyra_events_ratID' + str(rat_ID[rat_number]) + '.mat'
    data = sio.loadmat(data_path + data_file)
    complex_ripple = data[keywords[0]]
    ripple = data[keywords[1]]
    swr = data[keywords[2]]
    complex_ripple_count += complex_ripple.shape[0]
    ripple_count+=ripple.shape[0]
    swr_count+=swr.shape[0]
    count = count + complex_ripple.shape[0] + ripple.shape[0] + swr.shape[0]

complex_ripple_matrix= np.zeros((complex_ripple_count,len(time2)))
ripple_matrix = np.zeros((ripple_count,len(time2)))
swr_matrix = np.zeros((swr_count,len(time2)))
matrix = np.zeros((count,len(time2)))
class_vector = np.zeros((count,))
complex_ripple_count = 0
ripple_count = 0
swr_count = 0
count = 0
for rat_number in range(len(rat_ID)):
    data_file = 'HPCpyra_events_ratID' + str(rat_ID[rat_number]) + '.mat'
    data = sio.loadmat(data_path + data_file)
    complex_ripple = data[keywords[0]]
    ripple = data[keywords[1]]
    swr = data[keywords[2]]
    complex_ripple_matrix[complex_ripple_count:complex_ripple_count+complex_ripple.shape[0],:] = complex_ripple[:,time[time2]]
    ripple_matrix[ripple_count:ripple_count+ripple.shape[0],:] = ripple[:,time[time2]]
    swr_matrix[swr_count:swr_count+swr.shape[0],:] = swr[:,time[time2]]
    matrix[count:count+complex_ripple.shape[0],:] = complex_ripple[:,time[time2]]
    class_vector[count:count+complex_ripple.shape[0]]=0
    matrix[count + complex_ripple.shape[0]:count + complex_ripple.shape[0] + ripple.shape[0], :] = ripple[:,time[time2]]
    class_vector[count + complex_ripple.shape[0]:count + complex_ripple.shape[0] + ripple.shape[0]]=1
    matrix[count + complex_ripple.shape[0] + ripple.shape[0]:count + complex_ripple.shape[0] + ripple.shape[0] + swr.shape[0],:] = swr[:,time[time2]]
    class_vector[count + complex_ripple.shape[0] + ripple.shape[0]:count + complex_ripple.shape[0] + ripple.shape[0] + swr.shape[0]]=2
    count = count + complex_ripple.shape[0] + ripple.shape[0] + swr.shape[0]
    complex_ripple_count += complex_ripple.shape[0]
    ripple_count+=ripple.shape[0]
    swr_count+=swr.shape[0]

number_of_components = 5

pca1 = PCA(number_of_components)
pca1.fit(complex_ripple_matrix)
x = pca1.transform(complex_ripple_matrix)
transform1 = pca1.inverse_transform(x)

pca2 = PCA(number_of_components)
pca2.fit(ripple_matrix)
x = pca2.transform(ripple_matrix)
transform2 = pca2.inverse_transform(x)

pca3 = PCA(number_of_components)
pca3.fit(swr_matrix)
x = pca3.transform(swr_matrix)
transform3 = pca3.inverse_transform(x)

###save new_rat_data
for rat_number in range(len(rat_ID)):
    input_file_name = 'HPCpyra_events_ratID' + str(rat_ID[rat_number]) + '.mat'
    output_file_name = output_data_path + 'PCA_HPCpyra_events_ratID' + str(rat_ID[rat_number]) + '.mat'
    data = sio.loadmat(data_path + data_file)
    complex_ripple = data[keywords[0]][:,:3600]
    x = pca1.transform(complex_ripple)
    transform_CR = pca1.inverse_transform(x)
    ripple = data[keywords[1]][:,:3600]
    x = pca2.transform(ripple)
    transform_R = pca2.inverse_transform(x)
    swr = data[keywords[2]][:,:3600]
    x = pca3.transform(swr)
    transform_SWR = pca3.inverse_transform(x)
    output = {keywords[0]:transform_CR,keywords[1]:transform_R,keywords[2]:transform_SWR}
    savemat(output_file_name,output)

###save test data new_rat_data
for rat_number in range(len(rat_test)):
    input_file_name = 'HPCpyra_events_ratID' + str(rat_test[rat_number]) + '.mat'
    output_file_name = output_data_path + 'PCA_HPCpyra_events_ratID' + str(rat_test[rat_number]) + '.mat'
    data = sio.loadmat(data_path + data_file)
    complex_ripple = data[keywords[0]][:,:3600]
    x = pca1.transform(complex_ripple)
    transform_CR = pca1.inverse_transform(x)
    ripple = data[keywords[1]][:,:3600]
    x = pca2.transform(ripple)
    transform_R = pca2.inverse_transform(x)
    swr = data[keywords[2]][:,:3600]
    x = pca3.transform(swr)
    transform_SWR = pca3.inverse_transform(x)
    output = {keywords[0]:transform_CR,keywords[1]:transform_R,keywords[2]:transform_SWR}
    #np.save(output_file_name,output)
    savemat(output_file_name,output)


figure, axes = plt.subplots(1,2)
axes[0].set_title('ComplexRipple(CR)', fontsize=15)
axes[1].set_title('TEST CR Transformed', fontsize=15)
fs = 600
time = np.arange(1500, 2000)
time2 = np.arange(225, 375)
for i in range(50):

    min_value = np.min(complex_ripple[i, time[time2]])
    max_value = np.max(complex_ripple[i, time[time2]])
    norm_version = (complex_ripple[i, time[time2]] - min_value) / (max_value)
    axes[0].plot(time2 / fs, norm_version + i)

    min_value = np.min(transform_CR[i, time[time2]])
    max_value = np.max(transform_CR[i, time[time2]])
    norm_version = (transform_CR[i, time[time2]] - min_value) / (max_value)
    axes[1].plot(time2 / fs, norm_version + i)

figure.savefig(figure_path + 'transformed_test_complex_ripple_'+str(number_of_components)+'.png')
plt.show()

figure, axes = plt.subplots(1, 2)
axes[0].set_title('Ripple(R)', fontsize=15)
axes[1].set_title('TEST R Transformed', fontsize=15)
fs = 600
time = np.arange(1500, 2000)
time2 = np.arange(225, 375)
for i in range(50):
    min_value = np.min(ripple[i, time[time2]])
    max_value = np.max(ripple[i, time[time2]])
    norm_version = (ripple_matrix[i, time[time2]] - min_value) / (max_value)
    axes[0].plot(time2 / fs, norm_version + i)

    min_value = np.min(transform_R[i, time[time2]])
    max_value = np.max(transform_R[i, time[time2]])
    norm_version = (transform_R[i, time[time2]] - min_value) / (max_value)
    axes[1].plot(time2 / fs, norm_version + i)

figure.savefig(figure_path + 'transformed_test_ripple_' + str(number_of_components) + '.png')
plt.show()

figure, axes = plt.subplots(1, 2)
axes[0].set_title('SWR', fontsize=15)
axes[1].set_title('TEST SWR Transformed', fontsize=15)
fs = 600
time = np.arange(1500, 2000)
time2 = np.arange(225, 375)
for i in range(50):
    min_value = np.min(swr[i, time[time2]])
    max_value = np.max(swr[i, time[time2]])
    norm_version = (swr_matrix[i, time[time2]] - min_value) / (max_value)
    axes[0].plot(time2 / fs, norm_version + i)

    min_value = np.min(transform_SWR[i, time[time2]])
    max_value = np.max(transform_SWR[i, time[time2]])
    norm_version = (transform_SWR[i, time[time2]] - min_value) / (max_value)
    axes[1].plot(time2 / fs, norm_version + i)

figure.savefig(figure_path + 'transformed_test_SWR_' + str(number_of_components) + '.png')
plt.show()

#################################################################

figure = plt.figure()
gs = plt.GridSpec(1, 2)
axes0 = figure.add_subplot(gs[0, 0])
axes1 = figure.add_subplot(gs[0, 1])

f, t, Sxx = signal.spectrogram(swr[0,:], fs)
axes0.pcolormesh(t, f, Sxx, shading='gouraud')
# axes0.set_ylim([50, 200])
# axes0.set_xlim([2, 4])
axes0.set_ylabel('Frequency [Hz]', fontsize=15)
axes0.set_xlabel('Time [sec]', fontsize=15)
axes0.set_title('Original', fontsize=20)

f, t, Sxx = signal.spectrogram(transform_SWR[0,:], fs)
axes1.pcolormesh(t, f, Sxx, shading='gouraud')
# axes0.set_ylim([50, 200])
# axes0.set_xlim([2, 4])
axes1.set_ylabel('Frequency [Hz]', fontsize=15)
axes1.set_xlabel('Time [sec]', fontsize=15)
axes1.set_title('Transformed', fontsize=20)

figure.savefig(figure_path + 'transformed_example_testing_swr.png')
plt.show()

