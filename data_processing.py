##############################
#        SLURM stuff         #
##############################

import os
# Index for which of Anastasia's model to use
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])

# idx = 0


##############################
#           Preamble         #
##############################

import sys
sys.path.append('.')
import fluctuations
import twenty_one
import physics as phys

import numpy as np

import dill

########################################################
#    Import data from millicharged model               #
########################################################
data_folder = '/scratch/gpfs/hongwanl/millicharged_DM_with_bath/Vrel_scan_fixed_He_bug/for_Anastasia/'


mm_string_list = ['10_MeV', '30_MeV', '100_MeV', '300_MeV', '1_GeV', '3_GeV', '10_GeV', '30_GeV', '100_GeV', '300_GeV']

Vlis = np.loadtxt(open(data_folder+'Vlis.csv'), delimiter=',')
Qlis = np.loadtxt(open(data_folder+'Qlis.csv'), delimiter=',')
# zlis is in descending order. Flip to make data stored in positive z order. 
zlis = np.flipud(np.loadtxt(open(data_folder+'zlis.csv'), delimiter=','))

# in GeV
mmlis = np.array([0.01, 0.03, 0.1, 0.3, 1., 3., 10., 30., 100., 300.])

# dimensions mmlis x zlis x Qlis x Vlis
data = np.array([[np.loadtxt(data_folder+'mc100MeV_mm'+mm_string+'_z_'+str(int(z))+'.csv', delimiter=',') for z in zlis] for mm_string in mm_string_list])


# change to Vlis x Qlis x mmlis x zlis
data = np.transpose(data, axes=(3, 2, 0, 1)) / phys.kB


###############################################
#    Import data from Anastasia               #
###############################################

from scipy.io import loadmat

# number of models (140) x zlis
xA_data_full = loadmat('/scratch/gpfs/hongwanl/millicharged_DM_with_bath/xA_from_Anastasia/xa_astromodel.mat')['xout']

# number of models (140) x zlis
dTK_data_full = loadmat('/scratch/gpfs/hongwanl/millicharged_DM_with_bath/xA_from_Anastasia/dTK_astromodel.mat')['dTkout']

# narrow down to the data required by this job
xA_data  = xA_data_full[idx]
dTK_data = dTK_data_full[idx]


###############################################
#    Begin Data Processing                    #
###############################################

# add the heating dTK to the result for millicharged models
data = data + dTK_data

# actual T21 temperature
# dimensions Vlis x zlis x Qlis x mmlis
T21_data = np.moveaxis(twenty_one.T21(zlis, xA_data, data), -1, 1)

data_dict = {'z': zlis, 'Q': Qlis, 'm_m': mmlis}

# initialize the fluctuations class
T21_fluc = fluctuations.Fluctuations(Vlis*29., T21_data, data_dict=data_dict)

# initialize the power spectrum and the real space correlation function. 
_ = T21_fluc.Delta2_f()

# save the data
dir_name = '/scratch/gpfs/hongwanl/millicharged_DM_with_bath/power_spec/all_models/'

file_name = 'mc100MeV_full_smooth_deg3_'+str(idx)+'.p'

dill.dump(T21_fluc, open(dir_name+file_name, 'wb'))