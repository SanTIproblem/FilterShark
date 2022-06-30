import tensorlayer as tl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# lstm
def get_lstm_model(inputs_shape, masking_val):
    ni = tl.layers.Input(inputs_shape, name='input_layer')
    out = tl.layers.RNN(cell=tf.keras.layers.LSTMCell(units=64, recurrent_dropout=0.2),
                        return_last_output=True,
                        return_last_state=False,
                        return_seq_2d=True)(ni, sequence_length=tl.layers.retrieve_seq_length_op3(ni, pad_val=masking_val))
    nn = tl.layers.Dense(n_units=2, act=tf.nn.softmax, name="dense")(out)
    model = tl.models.Model(inputs=ni, outputs=nn, name='lstm')
    return model


# bilstm
def get_bilstm_model(inputs_shape, masking_val):
    ni = tl.layers.Input(inputs_shape, name='input_layer_2')
    # tensorlayer
    out = tl.layers.BiRNN(fw_cell=tf.keras.layers.LSTMCell(units=64, recurrent_dropout=0.2),
                          bw_cell=tf.keras.layers.LSTMCell(units=64, recurrent_dropout=0.2),
                          return_last_state=False,    
                          return_seq_2d=True)(ni, sequence_length=tl.layers.retrieve_seq_length_op3(ni, pad_val=masking_val))
    nn = tl.layers.Dense(n_units=2, act=tf.nn.softmax, name="dense_2")(out)
    model = tl.models.Model(inputs=ni, outputs=nn, name='bilstm')
    return model

