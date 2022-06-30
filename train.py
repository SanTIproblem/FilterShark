import logging
import math
import os
import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from sklearn.model_selection import train_test_split
import h5py
import json
from tensorflow.python.util import serialization
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras as keras
import datetime
from models.rnn import *
from models.rnn_attention import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def load_dataset(files, test_size=0.2):
    '''
    加载样本并取test_size的比例做测试集,valid_size的比例做验证集
    训练集：测试集：验证集=6：2：2
    总数据量大概19000条
    :param files:
    :param test_size:
    :return [x_train, y_train], [x_test, y_test] , [x_valid, y_valid]:
    '''
    x = []
    y = []

    # files = files[0:49]
    for file in files:
        data = np.load(file, allow_pickle=True)
        if (x == []) or (y == []):
            x = data['x']
            y = data['y']
        else:
            x = np.append(x, data['x'], axis=0)
            y = np.append(y, data['y'], axis=0)

    # 6:2:2
    x_train_valid, x_test, y_train_valid, y_test = train_test_split(x, y, test_size=test_size)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train_valid, y_train_valid, test_size=0.25)
    return x_train, y_train, x_test, y_test, x_valid, y_valid


def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)


def train(model, record_file):
    '''
    训练模型
    :param model:
    :return:
    '''

    # 超参数
    learning_rate = 1e-3
    n_epoch = 20
    batch_size = 128
    display_step = 50

    # Nadam优化器
    optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)

    # 评估指标 以及loss和accuracy
    y_p = []
    loss_vals = []
    acc_vals = []
    scores = {}
    scores.setdefault('fit_time', [])
    scores.setdefault('score_time', [])
    scores.setdefault('test_F1', [])
    scores.setdefault('test_Precision', [])
    scores.setdefault('test_Recall', [])
    scores.setdefault('test_Accuracy', [])
    scores.setdefault('test_Specificity', [])
    scores.setdefault('test_Sensitivity', [])
    scores.setdefault('test_AUC', [])

    logging.info("batch_size: %d", batch_size)
    logging.info("Start training the network...")
    record = []

    for epoch in range(n_epoch):
        step = 0
        total_step = math.ceil(len(x_train) / batch_size)

        # 利用训练集训练
        model.train()
        for batch_x, batch_y in tl.iterate.minibatches(x_train, y_train, batch_size, shuffle=True):

            start_time = time.time()
            # temp = copy.deepcopy(batch_x)
            max_seq_len = max([len(d) for d in batch_x])
            batch_y = batch_y.astype(np.int32)
            for i, d in enumerate(batch_x):
                batch_x[i] += [tf.convert_to_tensor(np.zeros(200), dtype=tf.float32) for i in
                               range(max_seq_len - len(d))]
                batch_x[i] = tf.convert_to_tensor(batch_x[i], dtype=tf.float32)
            batch_x = list(batch_x)
            batch_x = tf.convert_to_tensor(batch_x, dtype=tf.float32)
            # sequence_length = tl.layers.retrieve_seq_length_op3(batch_x, pad_val=masking_val)

            with tf.GradientTape() as tape:
                _y = model(batch_x)
                loss_val = tf.nn.sparse_softmax_cross_entropy_with_logits(batch_y, _y, name='train_loss')
                loss_val = tf.reduce_mean(loss_val)
            grad = tape.gradient(loss_val, model.trainable_weights)
            optimizer.apply_gradients(zip(grad, model.trainable_weights))

            loss_vals.append(loss_val)
            acc_vals.append(accuracy(_y, batch_y))

            if step + 1 == 1 or (step + 1) % display_step == 0:
                loss = sum(loss_vals) / len(loss_vals)
                acc = sum(acc_vals) / len(acc_vals)
                del loss_vals[:]
                del acc_vals[:]
                logging.info(
                    "Epoch {}/{},Step {}/{}, took {:.2f} s - train loss: {:.6f} - train accuracy: {:.5f}".format(
                        epoch + 1, n_epoch,
                        step, total_step,
                        time.time() - start_time,
                        loss, acc))
                record.append(
                    "Epoch {}/{},Step {}/{}, took {:.2f} s - train loss: {:.6f} - train accuracy: {:.5f}".format(
                        epoch + 1, n_epoch,
                        step, total_step,
                        time.time() - start_time,
                        loss, acc))
            step += 1

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_val.numpy(), step=epoch)
            tf.summary.scalar('accuracy', accuracy(_y, batch_y).numpy(), step=epoch)

        # 利用验证集评估
        model.eval()
        valid_loss, valid_acc, n_iter = 0, 0, 0
        for batch_x, batch_y in tl.iterate.minibatches(x_valid, y_valid, batch_size, shuffle=True):
            batch_y = batch_y.astype(np.int32)
            max_seq_len = max([len(d) for d in batch_x])
            for i, d in enumerate(batch_x):
                # 依照每个batch中最大样本长度将剩余样本打上padding
                batch_x[i] += [tf.convert_to_tensor(np.zeros(200), dtype=tf.float32) for i in
                               range(max_seq_len - len(d))]
                batch_x[i] = tf.convert_to_tensor(batch_x[i], dtype=tf.float32)
            batch_x = list(batch_x)
            batch_x = tf.convert_to_tensor(batch_x, dtype=tf.float32)

            _y = model(batch_x)

            loss_val = tf.nn.sparse_softmax_cross_entropy_with_logits(batch_y, _y, name='valid_loss')
            loss_val = tf.reduce_mean(loss_val)

            valid_loss += loss_val
            valid_acc += accuracy(_y, batch_y)
            n_iter += 1

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', loss_val.numpy(), step=epoch)
            tf.summary.scalar('accuracy', accuracy(_y, batch_y).numpy(), step=epoch)

        record.append("Epoch {}/{} - valid loss: {} - valid acc:  {} \n".format(epoch + 1, n_epoch,
                                                                                valid_loss / n_iter,
                                                                                valid_acc / n_iter))

    with open(record_file, 'a') as f:
        f.write('\n\n')
        for line in record:
            f.write(line + '\n')


def layer_conv1d_translator(tl_layer, _input_shape=None):
    args = tl_layer['args']
    name = args['name']
    filters = args['n_filter']
    kernel_size = [args['filter_size']]
    strides = [args['stride']]
    padding = args['padding']
    data_format = args['data_format']
    dilation_rate = [args['dilation_rate']]
    config = {'name': name, 'trainable': True, 'dtype': 'float32', 'filters': filters,
              'kernel_size': kernel_size, 'strides': strides, 'padding': padding, 'data_format': data_format,
              'dilation_rate': dilation_rate, 'activation': 'relu', 'use_bias': True,
              'kernel_initializer': {'class_name': 'GlorotUniform',
                                     'config': {'seed': None}
                                     },
              'bias_initializer': {'class_name': 'Zeros',
                                   'config': {}
                                   },
              'kernel_regularizer': None, 'bias_regularizer': None,
              'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}

    if _input_shape is not None:
        config['batch_input_shape'] = _input_shape
    result = {'class_name': 'Conv1D', 'config': config}
    return result


def layer_maxpooling1d_translator(tl_layer, _input_shape=None):
    args = tl_layer['args']
    name = args['name']
    pool_size = [args['filter_size']]
    strides = [args['strides']]
    padding = args['padding']
    data_format = args['data_format']
    config = {'name': name, 'trainable': True, 'dtype': 'float32', 'strides': strides, 'pool_size': pool_size,
              'padding': padding, 'data_format': data_format}
    if _input_shape is not None:
        config['batch_input_shape'] = _input_shape
    result = {'class_name': 'MaxPooling1D', 'config': config}
    return result


def layer_flatten_translator(tl_layer, _input_shape=None):
    args = tl_layer['args']
    name = args['name']

    config = {'name': name, 'trainable': True, 'dtype': 'float32', 'data_format': 'channels_last'}
    if _input_shape is not None:
        config['batch_input_shape'] = _input_shape
    result = {'class_name': 'Flatten', 'config': config}
    return result


def layer_dropout_translator(tl_layer, _input_shape=None):
    args = tl_layer['args']
    name = args['name']
    rate = 1 - args['keep']
    config = {'name': name, 'trainable': True, 'dtype': 'float32', 'rate': rate, 'noise_shape': None, 'seed': None}
    if _input_shape is not None:
        config['batch_input_shape'] = _input_shape
    result = {'class_name': 'Dropout', 'config': config}
    return result


def layer_dense_translator(tl_layer, _input_shape=None):
    args = tl_layer['args']
    name = args['name']
    units = args['n_units']
    config = {'name': name, 'trainable': True, 'dtype': 'float32', 'units': units, 'activation': 'softmax',
              'use_bias': True,
              'kernel_initializer': {'class_name': 'GlorotUniform', 'config': {'seed': None}},
              'bias_initializer': {'class_name': 'Zeros', 'config': {}},
              'kernel_regularizer': None,
              'bias_regularizer': None,
              'activity_regularizer': None,
              'kernel_constraint': None,
              'bias_constraint': None}

    if _input_shape is not None:
        config['batch_input_shape'] = _input_shape
    result = {'class_name': 'Dense', 'config': config}
    return result


def layer_rnn_translator(tl_layer, _input_shape=None):
    '''
    rnn层config转译方法
    :param tl_layer:
    :param _input_shape:
    :return:
    '''
    args = tl_layer['args']
    name = args['name']
    cell = {'class_name': 'LSTMCell', 'config': {'name': 'lstm_cell', 'trainable': True, 'dtype': 'float32',
                                                 'units': 64, 'activation': 'tanh', 'recurrent_activation': 'sigmoid',
                                                 'use_bias': True,
                                                 'kernel_initializer': {'class_name': 'GlorotUniform',
                                                                        'config': {'seed': None}},
                                                 'recurrent_initializer': {'class_name': 'Orthogonal',
                                                                           'config': {'gain': 1.0, 'seed': None}},
                                                 'bias_initializer': {'class_name': 'Zeros', 'config': {}},
                                                 'unit_forget_bias': True, 'kernel_regularizer': None,
                                                 'recurrent_regularizer': None, 'bias_regularizer': None,
                                                 'kernel_constraint': None, 'recurrent_constraint': None,
                                                 'bias_constraint': None, 'dropout': 0.0, 'recurrent_dropout': 0.2,
                                                 'implementation': 1}}
    config = {'name': name, 'trainable': True, 'dtype': 'float32', 'return_sequences': False,
              'return_state': False, 'go_backwards': False, 'stateful': False, 'unroll': False, 'time_major': False,
              'cell': cell
              }
    if _input_shape is not None:
        config['batch_input_shape'] = _input_shape
    result = {'class_name': 'RNN', 'config': config}
    return result


def layer_translator(tl_layer, is_first_layer=False):
    '''
    由于TensorLayer和keras的模型在保存配置信息config时，都是以layer为单位分别保存，
    因此在translate时，按照每个层的类型进行逐层转译
    :param tl_layer:
    :param is_first_layer:
    :return:
    '''
    _input_shape = None
    global input_shape
    if is_first_layer:
        _input_shape = input_shape
    if tl_layer['class'] == '_InputLayer':
        input_shape = tl_layer['args']['shape']
    elif tl_layer['class'] == 'Conv1d':
        return layer_conv1d_translator(tl_layer, _input_shape)
    elif tl_layer['class'] == 'MaxPool1d':
        return layer_maxpooling1d_translator(tl_layer, _input_shape)
    elif tl_layer['class'] == 'Flatten':
        return layer_flatten_translator(tl_layer, _input_shape)
    elif tl_layer['class'] == 'Dropout':
        return layer_dropout_translator(tl_layer, _input_shape)
    elif tl_layer['class'] == 'Dense':
        return layer_dense_translator(tl_layer, _input_shape)
    elif tl_layer['class'] == 'RNN':
        return layer_rnn_translator(tl_layer, _input_shape)
    return None


def config_translator(f_tl, f_k):
    '''
    转译模型配置信息config，包括模型结构、训练中loss metrics optimizer等，
    同时将信息传入masking层
    :param f_tl:
    :param f_k:
    :return:
    '''
    tl_model_config = f_tl.attrs['model_config'].decode('utf8')
    tl_model_config = eval(tl_model_config)
    tl_model_architecture = tl_model_config['model_architecture']

    k_layers = []

    masking_layer = {
        'class_name': 'Masking',
        'config': {
            'batch_input_shape': [None, None, 200],
            'dtype': 'float32',
            'mask_value': 0,
            'name': 'masking',
            'trainable': True
        }
    }
    k_layers.append(masking_layer)

    for key, tl_layer in enumerate(tl_model_architecture):
        if key == 1:
            k_layer = layer_translator(tl_layer, is_first_layer=True)
        else:
            k_layer = layer_translator(tl_layer)
        if k_layer is not None:
            k_layers.append(k_layer)
    f_k.attrs['model_config'] = json.dumps({'class_name': 'Sequential',
                                            'config': {'name': 'sequential', 'layers': k_layers},
                                            'build_input_shape': input_shape},
                                           default=serialization.get_json_type).encode('utf8')
    f_k.attrs['backend'] = keras.backend.backend().encode('utf8')
    f_k.attrs['keras_version'] = str(keras.__version__).encode('utf8')

    # todo: translate the 'training_config'
    training_config = {'loss': {'class_name': 'SparseCategoricalCrossentropy',
                                'config': {'reduction': 'auto', 'name': 'sparse_categorical_crossentropy',
                                           'from_logits': False}},
                       'metrics': ['accuracy'], 'weighted_metrics': None, 'loss_weights': None,
                       'sample_weight_mode': None,
                       'optimizer_config': {'class_name': 'Adam',
                                            'config': {'name': 'Adam', 'learning_rate': 0.01, 'decay': 0.0,
                                                       'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07,
                                                       'amsgrad': False
                                                       }
                                            }
                       }

    f_k.attrs['training_config'] = json.dumps(training_config, default=serialization.get_json_type).encode('utf8')


def weights_translator(f_tl, f_k):
    '''
    将训练中的权重（bias和kernel等）进行转译
    :param f_tl:
    :param f_k:
    :return:
    '''
    # todo: delete inputlayer
    if 'model_weights' not in f_k.keys():
        f_k_model_weights = f_k.create_group('model_weights')
    else:
        f_k_model_weights = f_k['model_weights']
    for key in f_tl.keys():
        if key not in f_k_model_weights.keys():
            f_k_model_weights.create_group(key)
        try:
            f_tl_para = f_tl[key][key]
        except KeyError:
            pass
        else:
            if key not in f_k_model_weights[key].keys():
                f_k_model_weights[key].create_group(key)
            weight_names = []
            f_k_para = f_k_model_weights[key][key]
            # todo：对RNN层的weights进行通用适配
            cell_name = ''
            if key == 'rnn_1':
                cell_name = 'lstm_cell'
                f_k_para.create_group(cell_name)
                f_k_para = f_k_para[cell_name]
                f_k_model_weights.create_group('masking')
                f_k_model_weights['masking'].attrs['weight_names'] = []
            for k in f_tl_para:
                if k == 'biases:0' or k == 'bias:0':
                    weight_name = 'bias:0'
                elif k == 'filters:0' or k == 'weights:0' or k == 'kernel:0':
                    weight_name = 'kernel:0'
                elif k == 'recurrent_kernel:0':
                    weight_name = 'recurrent_kernel:0'
                else:
                    raise Exception("cant find the parameter '{}' in tensorlayer".format(k))
                if weight_name in f_k_para:
                    del f_k_para[weight_name]
                f_k_para.create_dataset(name=weight_name, data=f_tl_para[k][:],
                                        shape=f_tl_para[k].shape)

        weight_names = []
        for weight_name in f_tl[key].attrs['weight_names']:
            weight_name = weight_name.decode('utf8')
            weight_name = weight_name.split('/')
            k = weight_name[-1]
            if k == 'biases:0' or k == 'bias:0':
                weight_name[-1] = 'bias:0'
            elif k == 'filters:0' or k == 'weights:0' or k == 'kernel:0':
                weight_name[-1] = 'kernel:0'
            elif k == 'recurrent_kernel:0':
                weight_name[-1] = 'recurrent_kernel:0'
            else:
                raise Exception("cant find the parameter '{}' in tensorlayer".format(k))
            if key == 'rnn_1':
                weight_name.insert(-1, 'lstm_cell')
            weight_name = '/'.join(weight_name)
            weight_names.append(weight_name.encode('utf8'))
        f_k_model_weights[key].attrs['weight_names'] = weight_names

    f_k_model_weights.attrs['backend'] = keras.backend.backend().encode('utf8')
    f_k_model_weights.attrs['keras_version'] = str(keras.__version__).encode('utf8')

    f_k_model_weights.attrs['layer_names'] = [i for i in f_tl.attrs['layer_names']]


def translator_tl2_keras_h5(_tl_h5_path, _keras_h5_path):
    f_tl_ = h5py.File(_tl_h5_path, 'r+')
    f_k_ = h5py.File(_keras_h5_path, 'a')
    f_k_.clear()
    weights_translator(f_tl_, f_k_)
    config_translator(f_tl_, f_k_)
    f_tl_.close()
    f_k_.close()


def format_convert(x, y):
    y = y.astype(np.int32)
    max_seq_len = max([len(d) for d in x])
    for i, d in enumerate(x):
        x[i] += [tf.convert_to_tensor(np.zeros(200), dtype=tf.float32) for i in range(max_seq_len - len(d))]
        x[i] = tf.convert_to_tensor(x[i], dtype=tf.float32)
    x = list(x)
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    return x, y


if __name__ == '__main__':

    masking_val = np.zeros(200)
    input_shape = None
    gradient_log_dir = 'logs/gradient_tape/'
    tensorboard = TensorBoard(log_dir=gradient_log_dir)

    # 定义log格式
    fmt = "%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO,
                        filename='train.log', filemode='a')

    # 加载数据
    x_train, y_train, x_test, y_test, x_valid, y_valid = load_dataset(
        ["./word2vec/output/sample_seq_pass.npz",
         "./word2vec/output/sample_seq_spam.npz"])

    # 载入模型（可选）
    model_opts = ['lstm', 'lstm_attention',
                  'bilstm', ]
    model = get_lstm_model(inputs_shape=[None, None, 200], masking_val=masking_val)
    # model = get_lstm_attention_model(inputs_shape=[None, None, 200], masking_val=masking_val)
    # model = get_bilstm_model(inputs_shape=[None, None, 200], masking_val=masking_val)

    for index, layer in enumerate(model.config['model_architecture']):
        if layer['class'] == 'RNN':
            if 'cell' in layer['args']:
                model.config['model_architecture'][index]['args']['cell'] = ''

    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    train_log_dir = gradient_log_dir + current_time + '/train'
    test_log_dir = gradient_log_dir + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    train_his = "./train_history/"
    if not os.path.exists(train_his):
        os.mkdir(train_his)
    record_file = train_his + f'{model_opts[0]}.txt'

    start_time = datetime.datetime.now()

    # 训练模型
    train(model, record_file)

    # 训练时间记录
    end_time = datetime.datetime.now()
    logging.info("Training Finished!")
    logging.info(f"Start Time: {start_time}...End Time: {end_time}...Total training time is {end_time - start_time}")

    # h5保存和转译
    model_dir = './model_h5'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    tl_h5_path = model_dir + f'/model_{model_opts[0]}_tl.hdf5'
    keras_h5_path = model_dir + f'/model_{model_opts[0]}_tl2k.hdf5'
    tl.files.save_hdf5_graph(network=model, filepath=tl_h5_path, save_weights=True)
    translator_tl2_keras_h5(tl_h5_path, keras_h5_path)

    # 评估模型
    new_model = keras.models.load_model(keras_h5_path)
    x_test, y_test = format_convert(x_test, y_test)
    score = new_model.evaluate(x_test, y_test, batch_size=128)

    # 保存SavedModel可部署文件
    saved_model_version = 1
    saved_model_path = f"./saved_models/{model_opts[0]}/"
    if os.path.isdir(saved_model_path) == False:
        logging.warning('Path (%s) not exists, making directories...', saved_model_path)
        os.makedirs(saved_model_path)
    tf.saved_model.save(new_model, saved_model_path + str(saved_model_version))
    logging.info("Saved Model!")
