import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

imshow_kwargs = {
    "interpolation": "none",
    "aspect": "auto",
    "vmin": -2,
    "vmax": 2,
}


""" Setting random seeds """
seed = 42

# TensorFlow
tf.random.set_seed(seed)

# Python
import random as python_random
python_random.seed(seed)

# NumPy (random number generator used for sampling operations)
rng = np.random.default_rng(seed)


class DataLoader(keras.utils.Sequence):

    def __init__(self, X, batch_size=16, batch_multiplier=10):
        
        # Data matrix
        self.X = X
        # Number of samples
        self.N_samples = X.shape[0]
        #self.N_samples = 1
        print('N_sample is:', self.N_samples)
        # Number of stations
        self.Nx = X.shape[1]  # channels
        # Number of time sampling points
        self.Nt = X.shape[2]  #time sample
        # Number of time sampling points for an input sample (fixed by architecture)
        self.win =  2048   # windows
        # Number of stations per batch sample
        self.N_sub = 50    #
        # Starting indices of the slices
        self.station_inds = np.arange(self.Nx - self.N_sub)
        # Batch size
        self.batch_size = batch_size
        self.batch_multiplier = batch_multiplier
        self.shuffle = True
        
        self.on_epoch_end()

    def __len__(self):
        """ Number of mini-batches per epoch """
        return int(self.batch_multiplier * self.N_samples * self.Nx *4 / float(self.batch_size * self.N_sub)-6)

    def on_epoch_end(self):
        """ Modify data """
        self.__data_generation()
        if self.shuffle == True:
            #ii = np.random.perm(len(self.X))
            #print(ii[0:32])
            #self.X = self.X[ii]
            np.random.shuffle(self.X)
            print('shuffled')
            plt.close("all")
            fig, axes = plt.subplots(ncols=2, figsize=(9, 4), constrained_layout=True, sharex="all", sharey="all")

            axes[0].imshow(self.masks[1,1,0:100,:], **imshow_kwargs)
            axes[0].set_title("Sample")
            axes[1].imshow(self.masked_samples[1,1,0:100,:] , **imshow_kwargs)
            axes[1].set_title("X")

        for ax in axes:
            ax.axis("off")

        plt.show()
            # Plot the matrix for the shuffling to check whether it is 
        #self.__data_repeat()
        return self.X
    
    def __data_repeat(self):
        final_packed_data = self.samples
        Nch = self.N_sub
        masked_data = []
        target_data = []
        # Loop over each event
        for event_idx in range(final_packed_data.shape[0]):
        # Get the data for the current event
            event_data = final_packed_data[event_idx]
            mask = np.ones((Nch, 2048))
            zero_line_idx = np.random.randint(0, Nch)  # Randomly select a row index
            mask[zero_line_idx] = 0  
            np.random.shuffle(mask)
        # Multiply the data for the current event with the mask
            masked_event_data = event_data * mask
            target = event_data * (1-mask)
        # Append the multiplied data for the current event to the list
            masked_data.append(masked_event_data)
            target_data.append(target)

        # Convert the list of masked data to a NumPy array
        masked_data = np.array(masked_data)
        target_data = np.array(target_data)
        self.X = final_packed_data
        self.masks = masked_data
        self.masked_samples = target_data
        pass 

    def __getitem__(self, idx):
        """ Select a mini-batch """
        batch_size = self.batch_size
        #selection = slice(idx * batch_size, (idx + 1) * batch_size) #= indexes in getitem
        #samples = np.expand_dims(self.samples[selection], -1)
        #samples[:,:,:,:]
        #masked_samples = np.expand_dims(self.masked_samples[selection], -1)
        #masks = np.expand_dims(self.masks[selection], -1)
        #print('samples shape')
        #print(samples)
        #print(samples.shape)
        #print('masked_samples shape')
        #print(masked_samples.shape)
        #print(masked_samples)
        #print('masks shape')
        #print(masks.shape)
        #print(masks)
        #list_IDs_temp = self.list_IDs[idx]
        samples = self.samples[idx]
        masked_samples = self.masked_samples[idx]
        masks = self.masks[idx]
        return (samples, masks), masked_samples
        #return masks,masked_samples

    def __data_generation(self):
        #No normalisation done 
        data_array = self.X
        N_samples = data_array.shape[0]
        Nch = data_array.shape[1]
        Nt = data_array.shape[2]
        N_sub = self.N_sub
        factor_2048 = 1
        reshaped_array = data_array[:, :, :2048]
        new_events_no = N_samples * factor_2048

        packed_data = []
        for event_idx in range(new_events_no):
            reshaped_temp = reshaped_array[:, ::-1, :]
            event_data = reshaped_temp[event_idx, :Nch, :]
            
            #Apply min-max normalization (Working!!!)
            #min_val = np.min(np.abs(event_data))
            #max_val = np.max(np.abs(event_data))
            #normalized_data = (np.abs(event_data) - min_val)/(max_val - min_val)
            
            #No Normalizations
            normalized_data = event_data
            
            num_blocks = Nch // N_sub
            for block_idx in range(num_blocks):
                start_idx = block_idx * N_sub
                end_idx = start_idx + N_sub
                block_data = normalized_data[start_idx:end_idx, :]
                packed_data.append(block_data)

        final_packed_data = np.array(packed_data)
        np.random.shuffle(final_packed_data)

        masked_data = []
        target_data = []
        new_final = []

        for event_idx in range(final_packed_data.shape[0]):
            event_data = final_packed_data[event_idx]
            mask1 = np.ones((N_sub, 2048))
            mask2 = np.ones((N_sub, 2048))
            mask3 = np.ones((N_sub, 2048))
            mask4 = np.ones((N_sub, 2048))
            #mask5 = np.ones((N_sub, 2048))
            zero_line_idx = np.random.randint(0, N_sub)
            one_line_idx = np.random.randint(0, N_sub)
            two_line_idx = np.random.randint(0, N_sub)
            three_line_idx = np.random.randint(0, N_sub)
            #four_line_idx = np.random.randint(0, N_sub)
            mask1[zero_line_idx] = 0
            mask2[one_line_idx] = 0
            mask3[two_line_idx] = 0
            mask4[three_line_idx] = 0
            #mask5[four_line_idx] = 0
            np.random.shuffle(mask1)
            np.random.shuffle(mask2)
            np.random.shuffle(mask3)
            np.random.shuffle(mask4)
            #np.random.shuffle(mask5)
            masked_event_data1 = event_data * mask1
            target1 = event_data * (1 - mask1)
            masked_data.append(masked_event_data1)
            target_data.append(target1)
            masked_event_data2 = event_data * mask2
            target2 = event_data * (1 - mask2)
            masked_data.append(masked_event_data2)
            target_data.append(target2)
            masked_event_data3 = event_data * mask3
            target3 = event_data * (1 - mask3)
            masked_data.append(masked_event_data3)
            target_data.append(target3)
            masked_event_data4 = event_data * mask4
            target4 = event_data * (1 - mask4)
            masked_data.append(masked_event_data4)
            target_data.append(target4)
            #masked_event_data5 = event_data * mask5
            #target5 = event_data * (1 - mask5)
            #masked_data.append(masked_event_data5)
            #target_data.append(target5)
            new_final.append(event_data)
            new_final.append(event_data)
            new_final.append(event_data)
            new_final.append(event_data)
            new_final.append(event_data)

        masked_data = np.array(masked_data)
        target_data = np.array(target_data)
        new_final =np.array(new_final)

        batch_size = self.batch_size
        num_batches = masked_data.shape[0] // batch_size
        old_num_batches = final_packed_data.shape[0] // batch_size
        print(old_num_batches)
        
        packed_new_final = new_final[:num_batches * batch_size].reshape(num_batches, batch_size, N_sub, 2048)
        packed_batches_masked = masked_data[:num_batches * batch_size].reshape(num_batches, batch_size, N_sub, 2048)
        packed_batches_target = target_data[:num_batches * batch_size].reshape(num_batches, batch_size, N_sub, 2048)
        packed_batches_final = final_packed_data[:old_num_batches * batch_size].reshape(old_num_batches, batch_size, N_sub, 2048)

        batched_data = []
        for i in range(num_batches):
            batched_data.append((packed_batches_masked[i], i))
            batched_data.append((packed_batches_target[i], i))
            batched_data.append((packed_new_final[i], i))# Append data with list ID

        for i in range(old_num_batches):
            batched_data.append((packed_batches_final[i], i))
            
        self.samples = packed_new_final
        self.masks = packed_batches_masked
        self.masked_samples = packed_batches_target
        print('samples shape:',packed_batches_target.shape )

        return packed_batches_masked, packed_batches_target
