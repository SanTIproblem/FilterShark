#! /usr/bin/env python3
"""
Word2vec训练词向量
"""

import collections
import logging
import os
import tarfile
import tensorflow as tf
import tensorlayer as tl
import tensorflow.keras as keras


def load_dataset():
    '''
    加载训练数据集
    数据集中词已经过jieba分词完毕
    :return 分词列表 words:
    '''
    prj = "https://github.com/tensorlayer/text-antispam"
    if not os.path.exists('data/msglog'):
        tl.files.maybe_download_and_extract(
            'msglog.tar.gz',
            'data',
            prj + '/raw/master/word2vec/data/')
        tarfile.open('data/msglog.tar.gz', 'r').extractall('data')
    files = ['data/msglog/msgpass.log.seg', 'data/msglog/msgspam.log.seg']
    words = []
    for file in files:
        f = open(file, encoding='utf-8')
        for line in f:
            for word in line.strip().split(' '):
                if word != '':
                    words.append(word)
        f.close()
    return words


def get_vocabulary_size(words, min_freq=3):
    size = 1  # 为UNK预留
    counts = collections.Counter(words).most_common()
    for word, c in counts:
        if c >= min_freq:
            size += 1
    return size


def save_weights(model, weights_file_path):
    path = os.path.dirname(os.path.abspath(weights_file_path))
    if os.path.isdir(path) == False:
        logging.warning('Path (%s) not exists, making directories...', path)
        os.makedirs(path)
    model.save_weights(filepath=weights_file_path)


def load_weights(model, weights_file_path):
    if os.path.isfile(weights_file_path):
        model.load_weights(filepath=weights_file_path)


def save_embedding(dictionary, network, embedding_file_path):
    '''
    将训练好的词向量保存到embedding_file_path.npy文件中
    :param dictionary:
    :param network:
    :param embedding_file_path:
    :return:
    '''
    words, ids = zip(*dictionary.items())
    params = network.normalized_embeddings
    embeddings = tf.nn.embedding_lookup(params, tf.constant(ids, dtype=tf.int32))
    # embeddings = tf.nn.embedding_lookup(params, tf.constant(ids, dtype=tf.int32)).eval()
    wv = dict(zip(words, embeddings))
    path = os.path.dirname(os.path.abspath(embedding_file_path))
    if os.path.isdir(path) == False:
        logging.warning('Path (%s) not exists, making directories...', path)
        os.makedirs(path)
    tl.files.save_any_to_npy(save_dict=wv, name=embedding_file_path + '.npy')


def train(model_name):
    '''
    训练词向量
    :param model_name:
    :return:
    '''

    # 超参数
    words = load_dataset()
    data_size = len(words)
    vocabulary_size = get_vocabulary_size(words, min_freq=3)
    batch_size = 500  # 一次Forword运算以及BP运算中所需要的训练样本数目
    embedding_size = 200  # 词向量维度
    skip_window = 5  # 上下文窗口，单词前后各取五个词
    num_skips = 10  # 从窗口中选取多少个预测对
    num_sampled = 64  # 负采样个数
    learning_rate = 0.025  # 学习率
    n_epoch = 50  # 所有样本重复训练50次
    num_steps = int((data_size / batch_size) * n_epoch)  # 总迭代次数

    data, count, dictionary, reverse_dictionary = tl.nlp.build_words_dataset(words, vocabulary_size)
    train_inputs = tl.layers.Input([batch_size], dtype=tf.int32)
    train_labels = tl.layers.Input([batch_size, 1], dtype=tf.int32)

    emb_net = tl.layers.Word2vecEmbedding(
        vocabulary_size=vocabulary_size,
        embedding_size=embedding_size,
        num_sampled=num_sampled,
        activate_nce_loss=True,
        nce_loss_args={})

    # embedding and loss
    emb, nce = emb_net([train_inputs, train_labels])
    model = tl.models.Model(inputs=[train_inputs, train_labels], outputs=[emb, nce])
    # model = tl.models.Model(inputs=[train_inputs, train_labels])
    # Adam优化器
    optimizer = keras.optimizers.Adam(learning_rate)

    # Start training
    model.train()
    weights_file_path = "weights/" + model_name + ".hdf5"
    load_weights(model, weights_file_path)

    loss_vals = []
    step = data_index = 0
    print_freq = 200
    while (step < num_steps):
        batch_inputs, batch_labels, data_index = tl.nlp.generate_skip_gram_batch(
            data=data, batch_size=batch_size, num_skips=num_skips,
            skip_window=skip_window, data_index=data_index)

        with tf.GradientTape() as tape:
            _, loss_val = model([batch_inputs, batch_labels])
        grad = tape.gradient(loss_val, model.trainable_weights)
        optimizer.apply_gradients(zip(grad, model.trainable_weights))

        loss_vals.append(loss_val)
        if step % print_freq == 0:
            logging.info("(%d/%d) latest average loss: %f.", step, num_steps, sum(loss_vals) / len(loss_vals))
            del loss_vals[:]
            save_weights(model, weights_file_path)
            embedding_file_path = "output/" + model_name
            save_embedding(dictionary, emb_net, embedding_file_path)
        step += 1


if __name__ == '__main__':
    fmt = "%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO,
                        filename='word2vec.log', filemode='a')

    train('model_word2vec_200')
