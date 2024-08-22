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
        self.N_sub = 11    #
        # Starting indices of the slices
        self.station_inds = np.arange(self.Nx - self.N_sub)
        # Batch size
        self.batch_size = batch_size
        self.batch_multiplier = batch_multiplier
        self.shuffle = True
        
        self.on_epoch_end()

    def __len__(self):
        """ Number of mini-batches per epoch """
        return int(self.batch_multiplier * self.N_samples * self.Nx / float(self.batch_size * self.N_sub))

    def on_epoch_end(self):
        """ Modify data """
        self.__data_generation()
        if self.shuffle == True:
            #ii = np.random.perm(len(self.X))
            #print(ii[0:32])
            #self.X = self.X[ii]
            np.random.shuffle(self.samples)
            print('shuffled')
            plt.close("all")
            fig, axes = plt.subplots(ncols=2, figsize=(9, 4), constrained_layout=True, sharex="all", sharey="all")

            axes[0].imshow(self.masks[1,0:11, :], **imshow_kwargs)
            axes[0].set_title("Sample")
            axes[1].imshow(self.masked_samples[1,0:11, :] , **imshow_kwargs)
            axes[1].set_title("X")

        for ax in axes:
            ax.axis("off")

        plt.show()
            # Plot the matrix for the shuffling to check whether it is 
        self.__data_repeat()
        pass
    
    def __data_repeat(self):
        final_packed_data = self.samples
        masked_data = []
        target_data = []
        # Loop over each event
        for event_idx in range(final_packed_data.shape[0]):
        # Get the data for the current event
            event_data = final_packed_data[event_idx]
            mask = np.ones((11, 2048))
            zero_line_idx = np.random.randint(0, 11)  # Randomly select a row index
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
        selection = slice(idx * batch_size, (idx + 1) * batch_size) #= indexes in getitem
        samples = np.expand_dims(self.samples[selection], -1)
        samples[:,:,:,:]
        masked_samples = np.expand_dims(self.masked_samples[selection], -1)
        masks = np.expand_dims(self.masks[selection], -1)
        #print('samples shape')
        #print(samples)
        #print(samples.shape)
        #print('masked_samples shape')
        #print(masked_samples.shape)
        #print(masked_samples)
        #print('masks shape')
        #print(masks.shape)
        #print(masks)
        
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

          # Find list of IDs
        #list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        #samples, masks, masked_samples = self.__data_generation(list_IDs_temp)
        #return (samples, masks), masked_samples
        return masks,masked_samples

    def __data_generation(self):
        # Reshape the format into (n,nch,2048)
        data_array = self.X
        N_samples = data_array.shape[0]
        Nch = data_array.shape[1]
        batch_size = self.batch_size
        Nt = data_array.shape[2]
        #factor_2048 = Nt // 2048
        factor_2048 = 1
        print(factor_2048)
        reshaped_array = data_array[:, :, :2048]
        #data_to_append = data_array[:, :, 2048:4096]
        #reshaped_array = array_sliced.reshape(N_samples*factor_2048, Nch, 2048)
        #reshaped_array = np.concatenate((array_sliced, data_to_append), axis=2)
        new_events_no = N_samples*factor_2048
        
        # Random Noise
        #noise  = rng.normal(size=(11, 2048))
        #data.shape
        #data = gaussian_filter(data, sigma=(1, 250))
        #noise = data.copy()
        
        #data = gaussian_filter(data, sigma=(1, 250))
        
        # Create one (11,2048)
        j = Nch // 11
        packed_data = []
        for event_idx in range(new_events_no):
            num_blocks = Nch // 11
            #Checking Point 1
            reshaped_temp = reshaped_array[:,::-1,:]
            event_data = reshaped_temp[event_idx,:num_blocks*11,:]
            # Initialize an empty list to store the packed data for the current event
            #event_packed_data = []
    
            # Loop over each block in the current event
            for block_idx in range(num_blocks):
            # Get the start and end indices of the current block
                start_idx = block_idx * 11
                end_idx = start_idx + 11
                #rng.shuffle(noise, axis=0)
        
            # Pack the current block into an array of shape (n, 11, 2048)
                block_data = event_data[start_idx:end_idx,:] 
                
                #Normalise within each block_data
                #Check 2: np.amax instead of max  --> by trace normalisation
                #np.amax(array[:])
                #block_data = block_data/np.amax(block_data[:])
                #event_packed_data.append(block_data)
                #print('Block data')
                #print(block_data.shape)
    
    # Append the packed data for the current event to the list
                packed_data.append(block_data)
            #print('event_packed_data')
            #print(event_packed_data.shape)

# Concatenate all the packed arrays to get the final packed data
        #np.random.shuffle(packed_data) Shuffle the list instead of np.array
        final_packed_data = np.array(packed_data)
        del reshaped_array
        #CHeck Print
        np.random.shuffle(final_packed_data)
        
#Now create masks
        masked_data = []
        target_data = []

# Loop over each event
        for event_idx in range(final_packed_data.shape[0]):
        # Get the data for the current event
            event_data = final_packed_data[event_idx]
            mask = np.ones((11, 2048))
            zero_line_idx = np.random.randint(0, 11)  # Randomly select a row index
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
    
        # Create one mini-batch
        num_batches = final_packed_data.shape[0] //batch_size
        #list_IDS = np.array([np.array(np.arange(1,num_batches)),np.array(np.arange(1,batch_size))])
        packed_batches_masked = masked_data[:num_batches * batch_size].reshape(num_batches, batch_size, 11, 2048)
        # = packed_batches_masked[0:num_batches, 0:batch_size]
        #print('This is list ID', list_IDS)
        #del masked_data
        packed_batches_target = target_data[:num_batches * batch_size].reshape(num_batches, batch_size, 11, 2048)
        #del target_data
        # Reshape final_packed_data into batches
        packed_batches_final = final_packed_data[:num_batches * batch_size].reshape(num_batches, batch_size, 11, 2048)
        #del final_packed_data
        
        self.samples = final_packed_data
        self.masks = masked_data
        self.masked_samples = target_data
        print('self samples shape')
        #print(samples)
        print(final_packed_data.shape)
        ##print('self masks shape')
        print(masked_data.shape)
        #self.list_IDS = list_IDS
        pass
        #return samples, masks, masked_data
