import jieba
import tensorlayer as tl
import numpy as np
from keras.models import load_model
from dl_filter.packages import text_regularization as tr


def text_tensor(text, wv):
    text = tr.extractWords(text)
    words = jieba.cut(text.strip())
    text_sequence = []
    for word in words:
        try:
            text_sequence.append(wv[word])
        except KeyError:
            text_sequence.append(wv['UNK'])
    text_sequence = np.asarray(text_sequence)
    sample = text_sequence.reshape(1, len(text_sequence), 200)
    return sample


def predict(text):
    wv = tl.files.load_npy_to_any(name='./dl_filter/model_word2vec_200.npy')
    model = load_model('./dl_filter/model_lstm_tl2k.hdf5')

    sample = text_tensor(text, wv)
    sample = np.array(sample, dtype=np.float32)
    len = sample.shape[1]
    sample = sample.reshape(len, 200)
    sample = sample.reshape(1, len, 200)
    data = sample.tolist()
    predictions = model.predict(data)
    print('result', predictions)
    result = np.argmax(predictions, axis=-1)
    print(type(result[0]),result[0])

    if result[0] == 0:
        return 0
    else:
        return 1
