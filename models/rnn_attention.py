import tensorlayer as tl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# lstm_attention
def get_lstm_attention_model(inputs_shape, masking_val, keep=0.2):
    '''
    构建模型
    :param inputs_shape:
    :return model:
    '''
    # size: (None, None, 200)
    ni = tl.layers.Input(inputs_shape, name='input_layer')
    # tensorlayer
    # size: (None, 64)
    out = tl.layers.RNN(cell=tf.keras.layers.LSTMCell(units=64, recurrent_dropout=0.2),
                        return_last_output=True,
                        return_last_state=False,
                        return_seq_2d=True)(ni,
                                            sequence_length=tl.layers.retrieve_seq_length_op3(ni, pad_val=masking_val))

    attention = layers.Attention()([ni, out])
    ni = layers.Concatenate()([ni, attention])

    nn = tl.layers.Dense(n_units=2, act=tf.nn.softmax, name="dense")(out)
    model = tl.models.Model(inputs=ni, outputs=nn, name='lstm_attention')
    return model