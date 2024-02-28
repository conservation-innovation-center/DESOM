"""
Implementation of the Deep Embedded Self-Organizing Map model
Autoencoder helper functions

@author Florent Forest
@version 2.0
"""

from keras.models import Model
from keras.layers import Layer, Concatenate, Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, BatchNormalization, LSTM
import numpy as np
from random import uniform

def mlp_autoencoder(encoder_dims,
                    act='relu',
                    init='glorot_uniform',
                    batchnorm=False):
    """Fully connected symmetric autoencoder model.

    Parameters
    ----------
    encoder_dims : list
        number of units in each layer of encoder. encoder_dims[0] is the input dim, encoder_dims[-1] is the
        size of the hidden layer (latent dim). The autoencoder is symmetric, so the total number of layers
        is 2*len(encoder_dims) - 1
    act : str (default='relu')
        activation of AE intermediate layers, not applied to Input, Hidden and Output layers
    init : str (default='glorot_uniform')
        initialization of AE layers
    batchnorm : bool (default=False)
        use batch normalization

    Returns
    -------
    ae_model, encoder_model, decoder_model : tuple
        autoencoder, encoder and decoder models
    """
    n_stacks = len(encoder_dims) - 1

    # Input
    x = Input(shape=(encoder_dims[0],), name='input')
    # Internal layers in encoder
    encoded = x
    for i in range(n_stacks-1):
        encoded = Dense(encoder_dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(encoded)
        if batchnorm:
            encoded = BatchNormalization()(encoded)
    # Hidden layer (latent space)
    encoded = Dense(encoder_dims[-1], activation='linear', kernel_initializer=init,
                    name='encoder_%d' % (n_stacks - 1))(encoded)  # latent representation is extracted from here
    # Internal layers in decoder
    decoded = encoded
    for i in range(n_stacks-1, 0, -1):
        decoded = Dense(encoder_dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(decoded)
        if batchnorm:
            decoded = BatchNormalization()(decoded)
    # Output
    decoded = Dense(encoder_dims[0], activation='linear', kernel_initializer=init, name='decoder_0')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=decoded, name='AE')

    # Encoder model
    encoder = Model(inputs=x, outputs=encoded, name='encoder')

    # Create input for decoder model
    encoded_input = Input(shape=(encoder_dims[-1],))
    # Internal layers in decoder
    decoded = encoded_input
    for i in range(n_stacks-1, -1, -1):
        decoded = autoencoder.get_layer('decoder_%d' % i)(decoded)
    # Decoder model
    decoder = Model(inputs=encoded_input, outputs=decoded, name='decoder')

    return autoencoder, encoder, decoder


def conv2d_autoencoder(input_shape,
                       latent_dim,
                       encoder_filters,
                       filter_size,
                       pooling_size,
                       act='relu',
                       init='glorot_uniform',
                       batchnorm=False):
    """2D convolutional autoencoder model.

    Parameters
    ----------
    input_shape : tuple
        input shape given as (height, width, channels) tuple
    latent_dim : int
        dimension of latent code (units in hidden dense layer)
    encoder_filters : list
        number of filters in each layer of encoder. The autoencoder is symmetric,
        so the total number of layers is 2*len(encoder_filters) - 1
    filter_size : int
        size of conv filters
    pooling_size : int
        size of maxpool filters
    act : str (default='relu')
        activation of AE intermediate layers, not applied to Input, Hidden and Output layers
    init : str (default='glorot_uniform')
        initialization of AE layers
    batchnorm : boolean (default=False)
        use batch normalization

    Returns
    -------
        ae_model, encoder_model, decoder_model : tuple
            autoencoder, encoder and decoder models
    """
    n_stacks = len(encoder_filters)

    # Infer code shape (assuming "same" padding, conv stride equal to 1 and max pooling stride equal to pooling_size)
    code_shape = list(input_shape)
    for _ in range(n_stacks):
        code_shape[0] = int(np.ceil(code_shape[0] / pooling_size))
        code_shape[1] = int(np.ceil(code_shape[1] / pooling_size))
    code_shape[2] = encoder_filters[-1]

    # Input
    x = Input(shape=input_shape, name='input')
    # Internal layers in encoder
    encoded = x
    for i in range(n_stacks):
        encoded = Conv2D(encoder_filters[i], filter_size, activation=act, padding='same', name='encoder_conv_%d'
                                                                                               % i)(encoded)
        if batchnorm:
            encoded = BatchNormalization()(encoded)
        encoded = MaxPooling2D(pooling_size, padding='same', name='encoder_maxpool_%d' % i)(encoded)
    # Flatten
    flattened = Flatten(name='flatten')(encoded)
    # Project using dense layer
    code = Dense(latent_dim, name='dense1')(flattened)  # latent representation is extracted from here
    # Project back to last feature map dimension
    reshaped = Dense(code_shape[0] * code_shape[1] * code_shape[2], name='dense2')(code)
    # Reshape
    reshaped = Reshape(code_shape, name='reshape')(reshaped)
    # Internal layers in decoder
    decoded = reshaped
    for i in range(n_stacks-1, -1, -1):
        if i > 0:
            decoded = Conv2D(encoder_filters[i], filter_size, activation=act, padding='same', name='decoder_conv_%d'
                                                                                                   % i)(decoded)
        else:
            decoded = Conv2D(encoder_filters[i], filter_size, activation=act, padding='valid', name='decoder_conv_%d'
                                                                                                    % i)(decoded)
        if batchnorm:
            decoded = BatchNormalization()(decoded)
        decoded = UpSampling2D(pooling_size, name='decoder_upsample_%d' % i)(decoded)
    # Output
    decoded = Conv2D(1, filter_size, activation='linear', padding='same', name='decoder_0')(decoded)

    # AE model
    autoencoder = Model(inputs=x, outputs=decoded, name='AE')

    # Encoder model (flattened output)
    encoder = Model(inputs=x, outputs=code, name='encoder')

    # Decoder model
    latent_input = Input(shape=(latent_dim,))
    flat_encoded_input = autoencoder.get_layer('dense2')(latent_input)
    encoded_input = autoencoder.get_layer('reshape')(flat_encoded_input)
    decoded = encoded_input
    for i in range(n_stacks-1, -1, -1):
        decoded = autoencoder.get_layer('decoder_conv_%d' % i)(decoded)
        decoded = autoencoder.get_layer('decoder_upsample_%d' % i)(decoded)
    decoded = autoencoder.get_layer('decoder_0')(decoded)
    decoder = Model(inputs=latent_input, outputs=decoded, name='decoder')

    return autoencoder, encoder, decoder

class seq2seqEncoder(Layer):
  """
  A keras Layer subclass used to encode time series data
  """
  def __init__(self, n_timesteps, n_features, n_hidden, name = 'seq2seqEncoder', **kwargs):
    """
    Parameters
    ---
    n_timesteps: int
        number of timesteps in input data
    n_features: int
        number of features in input data
    n_hidden: int
        number of hidden units in lstm layer
    """
    super().__init__(name = name, **kwargs)
    self.lstm = LSTM(units = n_hidden, input_shape = (n_timesteps, n_features), return_sequences = True, return_state = True)

  def call(self, input_seq):
    """
    Parameters
    ---
    input_seq: np.ndarray
        (B, T, N) input data array
    
    Return
    ---
    tensor (B, T, n_hidden): sequence outputs from lstm layer
    list[tensor (B, n_hidden), tensor (B, n_hidden)]: list of final hidden state and cell state
    """
    enc_outputs, enc_hidden_h, enc_hidden_s = self.lstm(input_seq)
    return enc_outputs, [enc_hidden_h, enc_hidden_s]

class seq2seqDecoder(Layer):
  """
  Keras Layer subclass for decoding 1D input into timeseries
  """
  def __init__(self, n_timesteps, n_hidden, n_features, teach_prob = 0.9, name = 'seq2seqDecoder', **kwargs):
    """
    Parameters
    ---
    n_timesteps: int
        number of time steps in output sequence
    n_hidden: int
        number of hidden features to create in decoder lstm
    n_dec_features: int
        number of features expected in incoming sequence data from encoder
    n_features: int
        number of features to output per observation
    teach_prob: float [0.0, 1.0]
        initial probability of teacher forcing - using target sequence data as input during training
    """

    super().__init__(name = name, **kwargs)
    self.dec_timesteps = n_timesteps
    self.n_hidden = n_hidden
    # self.n_enc_timesteps = n_enc_timesteps
    self.n_out_features = n_features
    self.lstm = LSTM(units = n_hidden, return_sequences = True, return_state = True)
    self.dense = Dense(n_features) # return 2 values, mu and logsigma (B, T, n_out_features)
    self.concatenate_features = Concatenate(axis = -1)
    self.concatenate_timesteps = Concatenate(axis = 1)
    self.teach_prob = teach_prob

  # def run_single_recurrent_step(self, enc_outputs, hidden):
  #   output, hidden = self.lstm(enc_outputs, hidden)

  #   return output, hidden
  def call(self, initial_input, hidden, targets, training = False, encoder_outputs = None):
    """
    Parameters
    ---
    initial_input: tensor
      observation at X_t
    hidden: tensor
      [hidden state h, hidden_state_c]
    targets: (B, T, N) tensor
      [X_t+1, X_t+2, X_t+3] target observations - only used for teacher forcing during training
    
    Return
    ---
    tensor: [B, n_timesteps, n_features] sequence of predicted outputs
    """
    decoder_sequence_length = self.dec_timesteps # the temporal dimension of tartets
    print('decoder sequence length', decoder_sequence_length)
    outputs = [None for _ in range(decoder_sequence_length)]
    input_at_t = initial_input
    print(f'teacher probability being used is {self.teach_prob}')
    for t in range(decoder_sequence_length):
      print(t)
    #   attention_vector = self.attention(current_hidden = hidden, encoder_outputs = encoder_outputs) # -> (B, 1, n_hidden)
    #   cat_input = self.concatenate_features([attention_vector, input_at_t])
      output, hidden_h, hidden_s = self.lstm(inputs = input_at_t, initial_state = hidden) # lstm models can take variable length inputs and will produce output of same length
      hidden = [hidden_h, hidden_s]
      outputs[t] = self.dense(output)

      teacher_force = uniform(0,1) < self.teach_prob if training else False

      # input_at_t = targets[:,t:t+1,:] if teacher_force else seq2seqDecoder.sample_gaussian(outputs[t])
      input_at_t = targets[:,t:t+1,:] if teacher_force else outputs[t]

    outputs = self.concatenate_timesteps(outputs)
    return outputs
   
def lstm_autoencoder(enc_timesteps, dec_timesteps, n_hidden, n_features):
    """
    Build a seq2seq autoencoder for time series data
    
    Parameters
    ---
    enc_timesteps: int
        number of timesteps in encoder input data
    dec_timesteps: int
        number of timesteps in decoder output data
    n_hidden: int
        number of hidden units in lstm encoder and decoder models
    n_features: int
        number of features in input and output data
    
    Returns
    ---
    tuple (Model, Model, Model): autoencoder, encoder, and decoder models
    """
    encoder_inputs = Input(shape = (enc_timesteps, n_features), name = 'encoder_inputs')
    decoder_targets = Input(shape = (dec_timesteps, n_features), name = 'decoder_targets')
    
    encoder_layers = seq2seqEncoder(n_timesteps = enc_timesteps, n_features = n_features, n_hidden = n_hidden)
    encoded, hidden = encoder_layers(encoder_inputs)
    
    decoder_layers = seq2seqDecoder(n_timesteps = dec_timesteps, n_hidden = n_hidden, n_features = n_features)
    decoded = decoder_layers(
        initial_input = encoder_inputs[:,-1:,], # initial input for decoder is the last of encoder input sequece
        hidden = hidden,
        targets = decoder_targets)
    
    # AE model
    autoencoder = Model(inputs=[encoder_inputs, decoder_targets], outputs=decoded, name='AE')

    # Encoder model
    encoder = Model(inputs = encoder_inputs, outputs = hidden, name = 'encoder')

    # Create input for decoder model - a placeholder for initial input and hidden state from encoder
    decoder_inputs = Input(shape = (n_features,), name = 'decoder_input')
    # Internal layers in decoder
    decoded = decoder_layers(initial_input = decoder_inputs, hidden = hidden)

    # Decoder model
    decoder = Model(inputs = decoder_inputs, outputs = decoded, name = 'decoder')
  
    return autoencoder, encoder, decoder
