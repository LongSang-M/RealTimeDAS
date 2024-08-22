import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,X, list_IDs , batch_size=32, N_mute = 11, n_events=32,
                 batch_multiplier = 3, shuffle=True):
        'Initialization'
        #Data Matrix
        self.x = x
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.list_IDs = list_IDs
        self.N_samples = X.shape[0]
        self.Nch = x.shape[1]
        self.Nt = x.shape[2]
        self.N_mute = N_mute
        #self.n_channels = n_channels
        #self.n_classes = n_classes
        self.n_events = n_events
        self.shuffle = shuffle
        self.batch_multiplier = batch_multiplier
        self.on_epoch_end()
        
    def __reshapedata__(self,X, N_mute):
        combined_array = self.x
        #Loop how many times per 4096 (32 events, 1039 channels )
        #for loop


    def __len__(self):
        'Denotes the number of batches per epoch' #Updated from jDAS calculation (Nx. N.sub, N_samples check the meaning)
        return int(self.batch_multiplier * self.N_samples * self.Nx / float(self.batch_size * self.N_sub))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the 
        # Generate samples, masks and target
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        # Reshape the format into (n,1029,2048)
        data_array = self.x
        N_samples = self.N_samples
        Nch = self.Nch
        batch_size = self.batch_size
        Nt = self.Nt
        factor_2048 = Nt // 2048
        array_sliced = data_array[:, :, :2048*factor_2048]
        reshaped_array = array_sliced.reshape(N_samples*factor_2048, Nch, 2048)
        new_nch = N_samples*factor_2048
        
        # Create one (11,2048)
        j = Nch // 11
        packed_data = []
        for event_idx in range(new_nch):
            num_blocks = Nch // 11
            event_data = reshaped_array[event_idx,-num_blocks*11:,:]
            # Initialize an empty list to store the packed data for the current event
            event_packed_data = []
    
            # Loop over each block in the current event
            for block_idx in range(num_blocks):
            # Get the start and end indices of the current block
                start_idx = block_idx * 11
                end_idx = start_idx + 11
        
            # Pack the current block into an array of shape (n, 11, 2048)
                block_data = event_data[start_idx:end_idx]
                event_packed_data.append(block_data)
    
    # Append the packed data for the current event to the list
            packed_data.extend(event_packed_data)

# Concatenate all the packed arrays to get the final packed data
        final_packed_data = np.array(packed_data)
        del reshaped_array
        np.random.shuffle(final_packed_data)
        
#Now create masks
        masked_data = []
        target_data = []

# Loop over each event
        for event_idx in range(final_packed_data.shape[0]):
        # Get the data for the current event
            event_data = final_packed_data[event_idx]
            mask = np.zeros((11, 2048))
            mask[:10, :] = 1
        # Shuffle the line of 0
            zero_line_idx = np.where(mask.sum(axis=1) == 0)[0][0]
            np.random.shuffle(mask[zero_line_idx])
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
        packed_batches_masked = masked_data[:num_batches * batch_size].reshape(num_batches, batch_size, 11, 2048)
        del masked_data
        packed_batches_target = target_data[:num_batches * batch_size].reshape(num_batches, batch_size, 11, 2048)
        del target_data
        # Reshape final_packed_data into batches
        packed_batches_final = final_packed_data[:num_batches * batch_size].reshape(num_batches, batch_size, 11, 2048)
        del final_packed_data
        
        self.samples = packed_batches_final
        self.masks = masked_data
        self.masked_samples = target_data
        pass
    
    
    
    def __data_generation(self):
        """ Generate a total batch """
        # Scale the data 
         #Test --> optimal 
        
        # Number of mini-batches
        N_batch = self.__len__()
        #print('N_batch is', N_batch)
        N_total = N_batch * self.batch_size
        #print('N total is ', N_total)
        # Buffer for mini-batches
        samples = np.zeros((N_total, self.N_sub, self.win))
        #print('samples is ', samples)
        # Buffer for masks
        masks = np.ones_like(samples)
        #print('mask is ', masks)
        
        batch_inds = np.arange(N_total)
        #print('batch_inds is ', batch_inds)
        np.random.shuffle(batch_inds)
        
        # Number of subsamples to create
        n_mini = N_total // self.N_samples
        #print('N total is ', N_total)
        
        # Loop over samples
        for s, sample in enumerate(self.X):
            # Random selection of station indices
            selection = rng.choice(self.station_inds, size=n_mini, replace=False)
            # Random time slice
            t_start = rng.integers(low=0, high=self.Nt-self.win)
            #t_start = rng.integers(low=0, high=self.Nt)
            t_slice = slice(t_start, t_start + self.win)
            # Time reversal
            order = rng.integers(low=0, high=2) * 2 - 1
            sign = rng.integers(low=0, high=2) * 2 - 1
            # Loop over station indices
            for k, station in enumerate(selection):
                # Selection of stations
                station_slice = slice(station, station + self.N_sub)
                subsample = sign * sample[station_slice, t_slice][:, ::order]
                # Get random index of this batch sample
                batch_ind = batch_inds[s * n_mini + k]
                # Store waveforms
                samples[batch_ind] = subsample
                # Select one waveform to blank
                blank_ind = rng.integers(low=0, high=self.N_sub)
                # Create mask
                masks[batch_ind, blank_ind] = 0
        
        print('self samples shape')
        #print(samples)
        print(samples.shape)
        print('self masks shape')
        print(masks.shape)
        #print(masks)        
        self.samples = samples
        self.masks = masks
        self.masked_samples = samples * (1 - masks)
        pass
    
    
    
    import numpy as np
from sklearn.preprocessing import MinMaxScaler

def __data_generation(self):
    # Extract data and dimensions
    data_array = self.X
    N_samples = data_array.shape[0]
    Nch = data_array.shape[1]
    Nt = data_array.shape[2]
    
    factor_2048 = 1
    reshaped_array = data_array[:, :, :2048]
    new_events_no = N_samples * factor_2048
    
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()
    
    packed_data = []
    for event_idx in range(new_events_no):
        reshaped_temp = reshaped_array[:, ::-1, :]
        event_data = reshaped_temp[event_idx, :Nch, :]
        
        # Reshape for min-max scaling
        reshaped_data = np.abs(event_data).reshape(-1, 1)
        
        # Fit and transform using MinMaxScaler
        scaler.fit(reshaped_data)
        normalized_data = scaler.transform(reshaped_data).reshape(Nch, Nt)
        
        # Divide into blocks
        num_blocks = Nch // 11
        for block_idx in range(num_blocks):
            start_idx = block_idx * 11
            end_idx = start_idx + 11
            block_data = normalized_data[start_idx:end_idx, :]
            packed_data.append(block_data)
    
    final_packed_data = np.array(packed_data)
    np.random.shuffle(final_packed_data)
    
    return final_packed_data