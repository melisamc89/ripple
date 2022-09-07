
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import signal

rat_ID = [3,4,9,201,203,206,210,211,213]
keywords = ['HPCpyra_complex_swr_veh','HPCpyra_ripple_veh','HPCpyra_swr_veh']
type_label = ['ComplexRipple','Ripple','SWR']
srate = 600
rat_number = 0

data_path = '/home/melisamc/Documentos/ripple/data/HPCpyra/'
figure_path = '/home/melisamc/Documentos/ripple/figures/'

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

class_list = []
class_list.append(complex_ripple_matrix)
class_list.append(ripple_matrix)
class_list.append(swr)


pca0 = PCA()
pca0.fit(matrix)

n_components = np.zeros((3,))
time_variable = np.arange(0,len(time2)) / 600
limit_variance = 0.85

figure = plt.figure()
gs = plt.GridSpec(3, 3)

axes0 = figure.add_subplot(gs[0:3, 0])#, projection='3d')
axes1 = figure.add_subplot(gs[0, 1])
axes2 = figure.add_subplot(gs[1, 1])
axes3 = figure.add_subplot(gs[2, 1])
axes4 = figure.add_subplot(gs[0, 2])
axes5 = figure.add_subplot(gs[1, 2])
axes6 = figure.add_subplot(gs[2, 2])

pca1 = PCA()
pca1.fit(complex_ripple_matrix)
ax0 = axes0.twinx()
axes0.scatter(np.arange(0,25),pca1.explained_variance_ratio_[0:25],color = color_list[0])
exp_variance_sum = np.cumsum(pca1.explained_variance_ratio_)
n_components[0] = np.where(exp_variance_sum>limit_variance)[0][0]
ax0.scatter(np.arange(0,25),exp_variance_sum[0:25],color = color_list[0],alpha = 0.5)

axes1.plot(time_variable,-pca1.components_[0,:],color = color_list[0])
axes2.plot(time_variable,pca1.components_[1,:],color = color_list[0])
axes3.plot(time_variable,pca1.components_[2,:],color = color_list[0])
axes4.plot(time_variable,pca1.components_[3,:],color = color_list[0])
axes5.plot(time_variable,pca1.components_[4,:],color = color_list[0])
axes6.plot(time_variable,pca1.components_[5,:],color = color_list[0])
axes0.axhline(y=pca1.explained_variance_ratio_[int(n_components[0])], color=color_list[0], linestyle='-',alpha=0.5)

pca2 = PCA()
pca2.fit(ripple_matrix)
axes0.scatter(np.arange(0,25),pca2.explained_variance_ratio_[0:25],color = color_list[1])
exp_variance_sum = np.cumsum(pca2.explained_variance_ratio_)
n_components[1] = np.where(exp_variance_sum>limit_variance)[0][0]
ax0.scatter(np.arange(0,25),exp_variance_sum[0:25],color = color_list[1],alpha = 0.5)
axes1.plot(time_variable,pca2.components_[0, :],color = color_list[1])
axes2.plot(time_variable,pca2.components_[1, :],color = color_list[1])
axes3.plot(time_variable,pca2.components_[2, :],color = color_list[1])
axes4.plot(time_variable,pca2.components_[3, :],color = color_list[1])
axes5.plot(time_variable,pca2.components_[4, :],color = color_list[1])
axes6.plot(time_variable,pca2.components_[5, :],color = color_list[1])
axes0.axhline(y=pca2.explained_variance_ratio_[int(n_components[1])], color=color_list[1], linestyle='-',alpha=0.5)

pca3 = PCA()
pca3.fit(swr_matrix)
axes0.scatter(np.arange(0,25),pca3.explained_variance_ratio_[0:25],color = color_list[2])
exp_variance_sum = np.cumsum(pca3.explained_variance_ratio_)
n_components[2] = np.where(exp_variance_sum>limit_variance)[0][0]
ax0.scatter(np.arange(0,25),exp_variance_sum[0:25],color = color_list[2],alpha = 0.5)
axes1.plot(time_variable,pca3.components_[0,:],color = color_list[2])
axes2.plot(time_variable,pca3.components_[1,:],color = color_list[2])
axes3.plot(time_variable,pca3.components_[2,:],color = color_list[2])
axes4.plot(time_variable,pca3.components_[3,:],color = color_list[2])
axes5.plot(time_variable,pca3.components_[4,:],color = color_list[2])
axes6.plot(time_variable,pca3.components_[5,:],color = color_list[2])
axes0.axhline(y=pca3.explained_variance_ratio_[int(n_components[2])], color=color_list[2], linestyle='-',alpha=0.5)


# axes0.scatter(np.arange(0,25),pca1.explained_variance_ratio_[0:25],color = 'k')
# exp_variance_sum = np.cumsum(pca1.explained_variance_ratio_)
# limit = pca1.explained_variance_ratio_[int(np.where(exp_variance_sum>limit_variance)[0][0])]
# axes0.axhline(y=limit, color='k', linestyle='-',alpha=0.5)
# axes1.plot(time_variable,pca.components_[0,:],color = 'k')
# axes2.plot(time_variable,pca.components_[1,:],color = 'k')
# axes3.plot(time_variable,pca.components_[2,:],color = 'k')
# axes4.plot(time_variable,pca.components_[3,:],color = 'k')
# axes5.plot(time_variable,pca.components_[4,:],color = 'k')
# axes6.plot(time_variable,pca.components_[5,:],color = 'k')
axes0.set_xlabel('Rank',fontsize = 20)
axes0.set_ylabel('Eigenvalue',fontsize = 20)
axes0.set_title('Eigenspectrum',fontsize = 25)
ax0.set_ylabel('Cumulative Variance')

axes1.set_title('Eigenvectors',fontsize = 25)

axes3.set_xlabel('Time (s)',fontsize = 20)
axes6.set_xlabel('Time (s)',fontsize = 20)

axes1.set_ylim([-0.5,0.5])
axes2.set_ylim([-0.5,0.5])
axes3.set_ylim([-0.5,0.5])
axes4.set_ylim([-0.5,0.5])
axes5.set_ylim([-0.5,0.5])
axes6.set_ylim([-0.5,0.5])
axes0.legend(['ComplexRipple','85%variance','Ripple','85%variance','SWR','85%variance'],fontsize = 15)
axes1.legend(['Direction1'])
axes2.legend(['Direction2'])
axes3.legend(['Direction3'])
axes4.legend(['Direction4'])
axes5.legend(['Direction5'])
axes6.legend(['Direction6'])

figure.set_size_inches([25,5])
figure.savefig(figure_path+'PCA_ripples_all.png')
plt.show()


###############################

figure = plt.figure()
gs = plt.GridSpec(3, 3)

axes0 = figure.add_subplot(gs[0:3, 0])#, projection='3d')
axes1 = figure.add_subplot(gs[0, 1])
axes2 = figure.add_subplot(gs[1, 1])
axes3 = figure.add_subplot(gs[2, 1])
axes4 = figure.add_subplot(gs[0, 2])
axes5 = figure.add_subplot(gs[1, 2])
axes6 = figure.add_subplot(gs[2, 2])
axes0.scatter(np.arange(0,25),pca0.explained_variance_ratio_[0:25],color = 'k')
exp_variance_sum = np.cumsum(pca0.explained_variance_ratio_)
limit = pca1.explained_variance_ratio_[int(np.where(exp_variance_sum>limit_variance)[0][0])]
axes0.axhline(y=limit, color='k', linestyle='-',alpha=0.5)
axes1.plot(time_variable,pca0.components_[0,:],color = 'k')
axes2.plot(time_variable,pca0.components_[1,:],color = 'k')
axes3.plot(time_variable,pca0.components_[2,:],color = 'k')
axes4.plot(time_variable,pca0.components_[3,:],color = 'k')
axes5.plot(time_variable,pca0.components_[4,:],color = 'k')
axes6.plot(time_variable,pca0.components_[5,:],color = 'k')

axes0.set_xlabel('Rank',fontsize = 20)
axes0.set_ylabel('Eigenvalue',fontsize = 20)
axes0.set_title('Eigenspectrum',fontsize = 25)
axes1.set_title('Eigenvectors',fontsize = 25)

axes3.set_xlabel('Time (s)',fontsize = 20)
axes6.set_xlabel('Time (s)',fontsize = 20)

axes1.set_ylim([-0.5,0.5])
axes2.set_ylim([-0.5,0.5])
axes3.set_ylim([-0.5,0.5])
axes4.set_ylim([-0.5,0.5])
axes5.set_ylim([-0.5,0.5])
axes6.set_ylim([-0.5,0.5])

axes1.legend(['Direction1'])
axes2.legend(['Direction2'])
axes3.legend(['Direction3'])
axes4.legend(['Direction4'])
axes5.legend(['Direction5'])
axes6.legend(['Direction6'])

figure.set_size_inches([15,5])
figure.savefig(figure_path+'PCA_all_ripples.png')
plt.show()
##########3
figure = plt.figure()
gs = plt.GridSpec(3, 3)

axes0 = figure.add_subplot(gs[0:3, 0])#, projection='3d')
axes1 = figure.add_subplot(gs[0:3, 1:3])

axes0.scatter(np.arange(0,25),pca0.explained_variance_ratio_[0:25],color = 'k')
exp_variance_sum = np.cumsum(pca0.explained_variance_ratio_)
limit = pca1.explained_variance_ratio_[int(np.where(exp_variance_sum>limit_variance)[0][0])]
axes0.axhline(y=limit, color='k', linestyle='-',alpha=0.5)

for i in range(2):
    axes1.plot(time_variable,pca0.components_[i,:])

axes0.set_xlabel('Rank',fontsize = 20)
axes0.set_ylabel('Eigenvalue',fontsize = 20)
axes0.set_title('Eigenspectrum',fontsize = 25)
axes1.set_title('Eigenvectors',fontsize = 25)

axes1.set_xlabel('Time (s)',fontsize = 20)

axes1.set_ylim([-0.5,0.5])

figure.set_size_inches([15,5])
figure.savefig(figure_path+'PCA_all_ripples_2.png')
plt.show()

##########################################################


X = pca0.transform(matrix)
figure = plt.figure()
gs = plt.GridSpec(1,1)
axes0 = figure.add_subplot(gs[0, 0], projection='3d')
#axes1 = figure.add_subplot(gs[0, 1], projection='3d')

for i in [0,1,2]:
    index = np.where(class_vector == i)[0]
    axes0.scatter(X[index,3],X[index,4], X[index,5],color = color_list[i])
    #axes1.scatter(X[index,3],X[index,4],X[index,5], color = color_list[i])

plt.show()

###############################################################3

fs = 600
for component in range(51):
    figure = plt.figure()
    gs = plt.GridSpec(2,2)
    axes0 = figure.add_subplot(gs[0, 0])
    axes1 = figure.add_subplot(gs[0, 1])
    axes2 = figure.add_subplot(gs[1, 0])
    axes3 = figure.add_subplot(gs[1, 1])

    f, t, Sxx = signal.spectrogram(pca0.components_[component,:], fs)
    a0 = axes0.pcolormesh(t, f, Sxx, shading='gouraud',vmin = 0,vmax = 0.0001)
    axes0.set_ylim([50,200])
    axes0.set_xlim([2,4])
    axes0.set_ylabel('Frequency [Hz]',fontsize = 15)
    axes0.set_xlabel('Time [sec]',fontsize = 15)
    axes0.set_title('ALL DATA',fontsize = 20)

    f, t, Sxx = signal.spectrogram(pca1.components_[component,:], fs)
    a1 = axes1.pcolormesh(t, f, Sxx, shading='gouraud',vmin = 0,vmax = 0.0001)
    axes1.set_ylim([50,200])
    axes1.set_xlim([2,4])
    axes1.set_ylabel('Frequency [Hz]',fontsize = 15)
    axes1.set_xlabel('Time [sec]',fontsize = 15)
    axes1.set_title('Complex Ripple',fontsize = 20)

    f, t, Sxx = signal.spectrogram(pca2.components_[component,:], fs)
    a2 = axes2.pcolormesh(t, f, Sxx, shading='gouraud',vmin = 0,vmax = 0.0001)
    axes2.set_ylim([50,200])
    axes2.set_xlim([2,4])
    axes2.set_ylabel('Frequency [Hz]',fontsize = 15)
    axes2.set_xlabel('Time [sec]',fontsize = 15)
    axes2.set_title('Ripple',fontsize = 20)

    f, t, Sxx = signal.spectrogram(pca3.components_[component,:], fs)
    a3 = axes3.pcolormesh(t, f, Sxx, shading='gouraud',vmin = 0,vmax = 0.0001)
    axes2.set_ylim([50,200])
    axes2.set_xlim([2,4])
    axes3.set_ylabel('Frequency [Hz]',fontsize = 15)
    axes3.set_xlabel('Time [sec]',fontsize = 15)
    axes3.set_title('SWR',fontsize = 20)

    figure.set_size_inches([15,10])
    figure.savefig(figure_path + 'spectrograms_component_' + str(component) + '.png')
    plt.show()

###########################################################################################################

number_of_components = 1

pca0 = PCA(number_of_components)
pca0.fit(matrix)

pca1 = PCA(number_of_components)
pca1.fit(complex_ripple_matrix)
x = pca1.transform(complex_ripple_matrix)
transform1 = pca1.inverse_transform(x)

time = np.arange(1500, 2000)
time2 = np.arange(225, 375)
figure, axes = plt.subplots(1,2)

for i in range(50):
    min_value = np.min(complex_ripple_matrix[i,time[time2]])
    max_value = np.max(complex_ripple_matrix[i,time[time2]])
    norm_version = (complex_ripple_matrix[i,time[time2]] - min_value)/(max_value)
    axes[0].plot(time2/fs,norm_version + i)

    min_value = np.min(transform1[i,time[time2]])
    max_value = np.max(transform1[i,time[time2]])
    norm_version = (transform1[i,time[time2]] - min_value)/(max_value)
    axes[1].plot(time2/fs,norm_version + i)

axes[0].set_title('ComplexRipple',fontsize = 15)
axes[1].set_title('TransformCR',fontsize = 15)
axes[0].set_xlabel('Time',fontsize = 12)
axes[1].set_xlabel('Time',fontsize = 12)
figure.savefig(figure_path + 'transformed_complex_ripple_'+str(number_of_components)+'.png')
plt.show()

pca2 =  PCA(number_of_components)
pca2.fit(ripple_matrix)
x = pca2.transform(ripple_matrix)
transform2 = pca2.inverse_transform(x)
figure, axes = plt.subplots(1,2)
for i in range(50):
    min_value = np.min(ripple_matrix[i,time[time2]])
    max_value = np.max(ripple_matrix[i,time[time2]])
    norm_version = (ripple_matrix[i,time[time2]] - min_value)/(max_value)
    axes[0].plot(time2/fs,norm_version + i)

    min_value = np.min(transform2[i,time[time2]])
    max_value = np.max(transform2[i,time[time2]])
    norm_version = (transform2[i,time[time2]] - min_value)/(max_value)
    axes[1].plot(time2/fs,norm_version + i)
axes[0].set_title('Ripple',fontsize = 15)
axes[1].set_title('TransformR',fontsize = 15)
axes[0].set_xlabel('Time',fontsize = 12)
axes[1].set_xlabel('Time',fontsize = 12)
figure.savefig(figure_path + 'transformed_ripple_'+str(number_of_components)+'.png')

plt.show()

pca3 = PCA(number_of_components)
pca3.fit(swr_matrix)
x = pca3.transform(swr_matrix)
transform3 = pca3.inverse_transform(x)
figure, axes = plt.subplots(1,2)
for i in range(50):
    min_value = np.min(ripple_matrix[i,time[time2]])
    max_value = np.max(ripple_matrix[i,time[time2]])
    norm_version = (swr_matrix[i,time[time2]] - min_value)/(max_value)
    axes[0].plot(time2/fs,norm_version + i)

    min_value = np.min(transform3[i,time[time2]])
    max_value = np.max(transform3[i,time[time2]])
    norm_version = (transform3[i,time[time2]] - min_value)/(max_value)
    axes[1].plot(time2/fs,norm_version + i)
axes[0].set_title('SWR',fontsize = 15)
axes[1].set_title('TransformSWR',fontsize = 15)
axes[0].set_xlabel('Time',fontsize = 12)
axes[1].set_xlabel('Time',fontsize = 12)
figure.savefig(figure_path + 'transformed_swr_'+str(number_of_components)+'.png')

plt.show()

########################################3
figure = plt.figure()
gs = plt.GridSpec(1, 2)
axes0 = figure.add_subplot(gs[0, 0])
axes1 = figure.add_subplot(gs[0, 1])

f, t, Sxx = signal.spectrogram(ripple_matrix[0,:], fs)
axes0.pcolormesh(t, f, Sxx, shading='gouraud')
# axes0.set_ylim([50, 200])
# axes0.set_xlim([2, 4])
axes0.set_ylabel('Frequency [Hz]', fontsize=15)
axes0.set_xlabel('Time [sec]', fontsize=15)
axes0.set_title('Original', fontsize=20)

f, t, Sxx = signal.spectrogram(transform2[0,:], fs)
axes1.pcolormesh(t, f, Sxx, shading='gouraud')
# axes0.set_ylim([50, 200])
# axes0.set_xlim([2, 4])
axes1.set_ylabel('Frequency [Hz]', fontsize=15)
axes1.set_xlabel('Time [sec]', fontsize=15)
axes1.set_title('Transformed', fontsize=20)

figure.savefig(figure_path + 'transformed_example_ripple.png')
plt.show()


######################################################################
figure, axes = plt.subplots(1,6)
axes[0].set_title('ComplexRipple', fontsize=15)
panel = 0
for number_of_components in [51,1,2,3,4,5]:

    pca1 = PCA(number_of_components)
    pca1.fit(swr_matrix)
    x = pca1.transform(swr_matrix)
    transform1 = pca1.inverse_transform(x)

    time = np.arange(1500, 2000)
    time2 = np.arange(225, 375)

    for i in range(50):
        min_value = np.min(transform1[i,time[time2]])
        max_value = np.max(transform1[i,time[time2]])
        norm_version = (transform1[i,time[time2]] - min_value)/(max_value)
        axes[panel].plot(time2/fs,norm_version + i)

    axes[panel].set_title('Nro: ' + str(number_of_components),fontsize = 15)
    axes[panel].set_xlabel('Time',fontsize = 12)
    panel = panel+1
figure.set_size_inches([25,10])
figure.savefig(figure_path + 'transformed_swr_'+str(number_of_components)+'.png')
plt.show()