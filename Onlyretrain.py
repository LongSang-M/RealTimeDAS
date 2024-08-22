import sys
import os
import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import shutil
import tensorflow as tf
import pickle
print(tf.__version__)

cwd = os.getcwd()
pardir = os.path.abspath(os.path.join(cwd, ".."))
modeldir = os.path.abspath(os.path.join(pardir, "models"))

if pardir not in sys.path:
    sys.path.append(pardir)

from jDAS import JDAS

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
imshow_kwargs = {
    "interpolation": "none",
    "aspect": "auto",
    #"vmin": -2,
    #"vmax": 2,
}

# Initialize an empty list to store arrays
arrays_list = []
# Setting the random seed
rng = np.random.default_rng(42)

for i in range(1, 46):
    file_path = f"./traindata_4096_SN/train_{i}.npy"
    # Load the array from the .npy file
    array = np.load(file_path)
    arrays_list.append(array)
arrays_combined = np.concatenate(arrays_list, axis=0)
temp = arrays_combined.T
X_scaler = MinMaxScaler()
X_scaler.fit(temp)
normalized_data = X_scaler.transform(temp)
X_std = normalized_data.T
clean_array = X_std.reshape(45,1029,4096)
del temp
del normalized_data
del X_std
del arrays_list
del arrays_combined
combined_array = clean_array
Nch, Nt = 1029, 45 * 4096
Nsplit = int(0.8 * 45)
combined_std = combined_array.std()
train_data_demo =combined_array[:Nsplit, :,: ]
#trained_flipped = train_data_demo[:,:,::-1]
#train_data_demo = np.concatenate((train_data_demo, trained_flipped), axis=0)
print(train_data_demo.shape)
#train_data_demo = train_data_demo / combined_array.std()
val_data_demo = combined_array[Nsplit:, :, :] 
#val_data_flipped = val_data_demo[:,:,::-1]
#val_data_demo = np.concatenate((val_data_demo, val_data_flipped), axis=0)
print(val_data_demo.shape)
#val_data_demo = val_data_demo / combined_array.std()
train_data = train_data_demo
val_data= val_data_demo
del train_data_demo
#del trained_flipped
del val_data_demo
#del val_data_flipped
jdas = JDAS()
model = jdas.load_model()
batch_size = 32
batch_multiplier =  1
train_loader = jdas.init_dataloader(train_data, batch_size, batch_multiplier)
val_loader = jdas.init_dataloader(val_data, batch_size, batch_multiplier)

from tensorflow.keras.callbacks import EarlyStopping

def train_and_save_history(model, model_name, train_loader, val_loader, epochs=30):
    logdir = os.path.join("logs", model_name)
    if os.path.isdir(logdir):
        shutil.rmtree(logdir)
    
    savefile = f"{model_name}.h5"
    savedir = os.path.join("save", model_name)
    if not os.path.isdir(savedir):
        os.makedirs(savedir)
    
    tensorboard_callback = jdas.callback_tensorboard(logdir)
    checkpoint_callback = jdas.callback_checkpoint(os.path.join(savedir, savefile))
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print(f"Training model: {model_name}")
    history = model.fit(train_loader, epochs=50, verbose=1, validation_data=val_loader,
                        callbacks=[tensorboard_callback, checkpoint_callback, early_stopping_callback], initial_epoch=0)
    print(f"Done training model: {model_name}")
    return history.history

history1 = train_and_save_history(model, "Relu_MinMax", train_loader, val_loader)
with open('history1.pkl', 'wb') as f:
    pickle.dump(history1, f)
##########
#Second Round
arrays_list = []

for i in range(1, 46):
    file_path = f"./traindata_4096_SN/train_{i}.npy"
    # Load the array from the .npy file
    array = np.load(file_path)
    arrays_list.append(array)
arrays_combined = np.concatenate(arrays_list, axis=0)
temp = arrays_combined.T
X_scaler = StandardScaler()
X_scaler.fit(temp)
normalized_data = X_scaler.transform(temp)
X_std = normalized_data.T
clean_array = X_std.reshape(45,1029,4096)
del temp
del normalized_data
del X_std
del arrays_list
del arrays_combined
combined_array = clean_array
Nch, Nt = 1029, 45 * 4096
Nsplit = int(0.8 * 45)
combined_std = combined_array.std()
train_data_demo =combined_array[:Nsplit, :,: ]
#trained_flipped = train_data_demo[:,:,:]
#train_data_demo = np.concatenate((train_data_demo, trained_flipped), axis=0)
print(train_data_demo.shape)
#train_data_demo = train_data_demo / combined_array.std()
val_data_demo = combined_array[Nsplit:, :, :] 
#val_data_flipped = val_data_demo[:,:,:]
#val_data_demo = np.concatenate((val_data_demo, val_data_flipped), axis=0)
print(val_data_demo.shape)
#val_data_demo = val_data_demo / combined_array.std()
train_data = train_data_demo
val_data= val_data_demo
del train_data_demo
#del trained_flipped
del val_data_demo
#del val_data_flipped
jdas = JDAS()
model = jdas.load_model()
batch_size = 32
batch_multiplier =  1
train_loader = jdas.init_dataloader(train_data, batch_size, batch_multiplier)
val_loader = jdas.init_dataloader(val_data, batch_size, batch_multiplier)
#Second Round
history2 = train_and_save_history(model, "Relu_SoftS", train_loader, val_loader)
with open('history2.pkl', 'wb') as f:
    pickle.dump(history2, f)

# Third Round

########################################


#history3 = train_and_save_history(model, "SoftS", train_loader, val_loader)
#with open('history3.pkl', 'wb') as f:
#    pickle.dump(history3, f)

### Plotting
#with open('history1.pkl', 'wb') as f:
#    pickle.dump(history1, f)
#with open('history2.pkl', 'wb') as f:
#    pickle.dump(history2, f)
#with open('history3.pkl', 'wb') as f:
#    pickle.dump(history3, f)

# Load histories
with open('history1.pkl', 'rb') as f:
    history1 = pickle.load(f)
with open('history2.pkl', 'rb') as f:
    history2 = pickle.load(f)
with open('Min1.pkl', 'rb') as f:
    history3 = pickle.load(f)

# Plot training loss for each model
plt.figure()
plt.plot(history1['loss'])
plt.plot(history2['loss'])
plt.plot(history3['loss'])
plt.title('Training Loss Comparison - Number of Mask')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['1 Mask', '2 Masks', '3 Masks'])
plt.show()

# Plot validation loss for each model
plt.figure()
plt.plot(history1['val_loss'])
plt.plot(history2['val_loss'])
plt.plot(history3['val_loss'])
plt.title('Validation Loss Comparison - Number of Mask')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['1 Mask', '2 Masks', '3 Masks'])
plt.show()


