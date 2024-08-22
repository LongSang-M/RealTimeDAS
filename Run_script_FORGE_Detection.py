import time
import sys
import os
sys.path.append('./das/')
from Detector import Detector
import obspy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable



#%%
#define output directory for results

################    DAS object and filtering parameters   ##############################
path_input = './Data/detectiondata/'
#fname = 'FORGE_DFIT_UTC_20220421_135024.398.tdms'
fname = 'FORGE_DFIT_UTC_20220421_160009.398.tdms'
fname2 = 'FORGE_DFIT_UTC_20220421_160024.398.tdms'
#fname = 'receivers.h5'
#fname = 'synthetic_seismogram_500_4500_DAS_4000.npy'
f = path_input + fname
file_format = 'tdms'

dx = 1.02
gl = 10.
downsampling_rate = 1000
ftype = 'bandpass'
freqmin, freqmax = 10, 249
k0 = True
low_vel_events = True

xini=1350
xend=2400
#xini=250
#xend=1160
tini=0
tend=15


################    Clustering parameters   ##############################

min_numb_detections=10
max_dist_detections=2

################    SEMBLANCE parameters   ##############################

ns_window = 20 #width of data window for semblance
a = 20
b = 375
b_step =  10  # db sample step along time 

c_min, c_max, c_step = 70, 500, 5 # set of parameters for characterization of the hyperbole (for semblance)

d_min, d_max, d_step = 0, 1200, 50 #dd sample step along distance, dmax maximum step along distance

d_static = 892

#svd singular value decomposition for weighting for the semblance matrix 
svd = False

# apply semblance analysis also along distance
lat_search = False
savefig = False



#%%
t0=time.time()

## Read and visualize input file
arr = Detector(f, dx, gl, fname, file_format); #arr.visualization(arr.traces, f, savefig=True)

## Select a subset of the data 
arr.data_select(endtime = tend,
                startlength = xini,
                endlength = xend); #arr.visualization(arr.traces, f)

# Denoise the selected data
arr.denoise(data = arr.traces,
            sampling_rate_new = downsampling_rate,
            ftype = ftype,
            fmin = freqmin,
            fmax = freqmax,
            k0=k0,
            low_vel_events = low_vel_events); #arr.visualization(arr.traces, f)
n_ratio_reshaped = arr.n_ratio


## Apply the detector
arr.detector(ns_window = ns_window,
              a = a,
              b_step = b_step,
              c_min = c_min,
              c_max = c_max,
              c_step = c_step,
              d_static = d_static,
              d_min = d_min,
              d_max = d_max,
              d_step = d_step,
              svd = svd,
              lat_search = lat_search)

print('Total time is {} seconds'.format(time.time()-t0))

## PLOT RESULTS
#events = arr.detected_events(min_numb_detections,max_dist_detections)
#arr.plot(arr.traces, tini, tend, fname, b_step, savefig=savefig)
#arr.plotfk(savefig=False)

#%%
################    SEMBLANCE parameters tuning   ##############################

#if lat_search == False:
#    arr.hyperbolae_tuning(a, b, d_static, c_min, c_max, c_step)

#else:
#    arr.hyperbolae_tuning(a, b, arr.d_best, c_min, c_max, c_step)
    

percentile_threshold = 99.7
percentile_value = np.percentile(np.absolute(arr.traces), percentile_threshold)
#personalized_traces = np.where(arr.traces > percentile_value, 1, arr.traces)
clipped_traces = np.clip(arr.traces, -percentile_value, +percentile_value)
n_ratio_reshaped = np.expand_dims(n_ratio_reshaped,-1)
data_all = clipped_traces*n_ratio_reshaped
fname = fname2
#fname = 'FORGE_DFIT_UTC_20220421_160009.398.tdms'
#fname = 'receivers.h5'
#fname = 'synthetic_seismogram_500_4500_DAS_4000.npy'
f = path_input + fname
file_format = 'tdms'

dx = 1.02
gl = 10.
downsampling_rate = 1000
ftype = 'bandpass'
freqmin, freqmax = 10, 249
k0 = True
low_vel_events = True

xini=1350
xend=2400
#xini=250
#xend=1160
tini=0
tend=15


################    Clustering parameters   ##############################

min_numb_detections=10
max_dist_detections=2

################    SEMBLANCE parameters   ##############################

ns_window = 20 #width of data window for semblance
a = 20
b = 375
b_step =  10  # db sample step along time 

c_min, c_max, c_step = 70, 500, 5 # set of parameters for characterization of the hyperbole (for semblance)

d_min, d_max, d_step = 0, 1200, 50 #dd sample step along distance, dmax maximum step along distance

d_static = 892

#svd singular value decomposition for weighting for the semblance matrix 
svd = False

# apply semblance analysis also along distance
lat_search = False
savefig = False



#%%
t0=time.time()

## Read and visualize input file
arr = Detector(f, dx, gl, fname, file_format); #arr.visualization(arr.traces, f, savefig=True)

## Select a subset of the data 
arr.data_select(endtime = tend,
                startlength = xini,
                endlength = xend); #arr.visualization(arr.traces, f)

# Denoise the selected data
arr.denoise(data = arr.traces,
            sampling_rate_new = downsampling_rate,
            ftype = ftype,
            fmin = freqmin,
            fmax = freqmax,
            k0=k0,
            low_vel_events = low_vel_events); #arr.visualization(arr.traces, f)
n_ratio_reshaped1 = arr.n_ratio


## Apply the detector
arr.detector(ns_window = ns_window,
              a = a,
              b_step = b_step,
              c_min = c_min,
              c_max = c_max,
              c_step = c_step,
              d_static = d_static,
              d_min = d_min,
              d_max = d_max,
              d_step = d_step,
              svd = svd,
              lat_search = lat_search)

print('Total time is {} seconds'.format(time.time()-t0))

## PLOT RESULTS
#events = arr.detected_events(min_numb_detections,max_dist_detections)
#arr.plot(arr.traces, tini, tend, fname, b_step, savefig=savefig)
#arr.plotfk(savefig=False)

#%%
################    SEMBLANCE parameters tuning   ##############################

#if lat_search == False:
#    arr.hyperbolae_tuning(a, b, d_static, c_min, c_max, c_step)

#else:
#    arr.hyperbolae_tuning(a, b, arr.d_best, c_min, c_max, c_step)
    

percentile_threshold = 99.7
percentile_value = np.percentile(np.absolute(arr.traces), percentile_threshold)
#personalized_traces = np.where(arr.traces > percentile_value, 1, arr.traces)
clipped_traces1 = np.clip(arr.traces, -percentile_value, +percentile_value)
n_ratio_reshaped1 = np.expand_dims(n_ratio_reshaped1,-1)
data_all_noise = clipped_traces1*n_ratio_reshaped1

import shutil
import tensorflow as tf
print(tf.__version__)

cwd = os.getcwd()
pardir = os.path.abspath(os.path.join(cwd, "Extralib"))
modeldir = os.path.abspath(os.path.join(pardir, "models"))

if pardir not in sys.path:
    sys.path.append(pardir)

from jDAS import JDAS

jdas = JDAS()
model = jdas.load_model()

""" Callbacks """

model_name = "51_MinMax"

logdir = os.path.join("logs", model_name)
if os.path.isdir(logdir):
    shutil.rmtree(logdir)
    
savefile = "51_MinMax.h5"
savedir = os.path.join("save", model_name)
if not os.path.isdir(savedir):
    os.makedirs(savedir)

tensorboard_callback = jdas.callback_tensorboard(logdir)
checkpoint_callback = jdas.callback_checkpoint(os.path.join(savedir, savefile))
model = jdas.load_model(os.path.join(savedir, savefile))
post_train = jdas.denoise(data_all)
#data_all_noise = np.load('./whitenoise.npy')
post_train2= jdas.denoise(data_all_noise)

percentile_threshold = 99.7
percentile_value = np.percentile(np.absolute(post_train), percentile_threshold)

fig, ((ax1)) = plt.subplots(1, 1,  figsize=(10, 4), sharex=True, constrained_layout=False, gridspec_kw={'height_ratios': [1]})

# Plotting
im3 = ax1.imshow((post_train[:,:]) , cmap='seismic', aspect='auto' , clim = [-percentile_value,percentile_value])
ax1.set_ylabel('Distance to the well [m]', fontsize=10)
ax1.set_xlabel('Relative time [s]', fontsize=10)
ax1.set_title('Denoised Result', fontsize=12)  # Adding title
#ax1.set_xticks([334, 334*2, 334*3, 334*4, 334*5])
#ax1.set_xticks([1000, 2000, 3000, 4000])
#ax1.set_xticklabels([1, 2, 3, 4])

divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='1.4%', pad=.05)
fig.colorbar(im3, cax=cax, orientation='vertical')
cax.xaxis.set_ticks_position("none")

# Optionally, add plotting for ax1 and ax2 if needed
# Remember to adjust the subplot layout accordingly

plt.show()


from obspy import signal
from obspy.signal.trigger import classic_sta_lta, recursive_sta_lta
from obspy.signal.trigger import plot_trigger
from obspy import Trace
df = 2000
#data_all = clipped_traces*n_ratio_reshaped
sampling_rate = 1000  # samples per second
sta_win = 0.3  # seconds
lta_win = 10  # seconds
on_threshold = 4.0
off_threshold = 3.8

# Traces to select
trace_indices = [520, 570,  670, 720, 770, 820, 870]
num_traces = len(trace_indices)
trigger_count_threshold = 4  # At least 4 traces should be triggered

# Store results
trigger_counts = 0
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Assuming you have loaded your DAS data into a variable named 'data'
# Replace this with your actual data loading code

# Define the bandpass filter parameters
fs = 1000  # Sampling frequency of your DAS data
lowcut = 1  # Low cutoff frequency (in Hz)
highcut = 250  # High cutoff frequency (in Hz)
order = 4  # Filter order

# Function to apply bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# Apply bandpass filter to your DAS data
data_all_noise = butter_bandpass_filter(post_train2, 10, 150, 1000, 2)
data_all= butter_bandpass_filter(post_train, 10, 150, 1000, 2)
n=0.3
# Process each selected trace
for trace_idx in trace_indices:
    # Convert data to ObsPy Trace object
    trace_data = data_all[trace_idx]
    temp = np.concatenate((n*data_all_noise[trace_idx], trace_data))
    trace = Trace(data=temp)
    trace.stats.sampling_rate = sampling_rate
    n = n + 0.05
    # Compute STA/LTA ratio
    cft = recursive_sta_lta(trace.data, int(sta_win * sampling_rate), int(lta_win * sampling_rate))
    
    # Detect events
    events = np.where(cft > on_threshold)[0]
    
    # Check if any events are detected
    if len(events) > 0:
        trigger_counts += 1
    
    # Plotting the results
    plt.figure(figsize=(12, 6))
    plot_trigger(trace, cft, on_threshold, off_threshold)
    plt.title(f'STA/LTA Event Detection for Trace {trace_idx}')
    plt.xlim(-15 * sampling_rate, 15 * sampling_rate)  # Set x-axis limits to [-15, 15] seconds
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

# Check if at least 4 traces have been triggered
if trigger_counts >= trigger_count_threshold:
    print("An event has been detected based on the criteria.")
else:
    print("No event detected based on the criteria.")