import sys
from pathlib import Path
import numpy as np
DIR = Path().resolve()
sys.path.append(str(DIR/'DESOM'))

from tensorflow.keras.layers import ReLU

# from SOM import SOMLayer
# from ConvDESOM import ConvDESOM
import AE  
import DESOM

# create dummy data for testing models
test_county_dat = np.random.uniform(low = 0, high = 1, size = (500, 4, 15))
county_dat_train = test_county_dat[0:(500*3)//4,:]
county_dat_val = test_county_dat[(500*3)//4:,:]
test_block_dat = np.random.uniform(low = 0, high = 1, size = (5000, 4, 25))
block_dat_train = test_block_dat[0:(5000*3)//4,:]
block_dat_val = test_block_dat[(5000*3)//4:,:]

# create and initialize DESOM models for county and block data
county_desom = DESOM.DESOM(encoder_dims = [test_county_dat.shape[-1],32,64,128,256,128,64,32], map_size = (10,10))
county_desom.initialize()
block_desom = DESOM.DESOM(encoder_dims = [test_block_dat.shape[-1],32,64,128,256,128,64,32], map_size = (10,10))
block_desom.initialize()

batch = 20
epochs = 5
iterations = (len(county_dat_train)//batch) * 5
county_desom.pretrain(test_county_dat[:,0,:], epochs = 100, batch_size = 20, save_dir = './')
county_desom.compile(gamma = 1.0)
county_desom.fit(X_train = county_dat_train[:,0,:], X_val = county_dat_val[:,0,:], batch_size = batch, iterations = iterations, save_dir = './')

# lstm autoencoder 
autoencoder, encoder, decoder = AE.lstm_autoencoder(
    input_dim = [4, 15], # t, nvaribles
    n_latent = 32,
    activation = ReLU(max_value = 2.0))

county_temporal_desom = DESOM.LstmDESOM(
    hidden = 32,
    input_dim = [4,15],
    encoder_dims = [1],
    map_size = (10,10))

county_temporal_desom.initialize()
county_temporal_desom.pretrain(test_county_dat, epochs = 10, batch_size = 20, save_dir = './')
county_temporal_desom.compile(gamma = 1.0)
county_temporal_desom.fit(X_train = county_dat_train, X_val = county_dat_val, batch_size = batch, iterations = iterations, save_dir = './')

