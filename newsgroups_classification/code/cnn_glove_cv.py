# python3
# -*- coding: utf-8 -*-
# @Author  : lina
# @Time    : 2018/6/10 15:31
'''
使用CNN进行文本分类，1000个文档分成20类，五重交叉验证结果
'''

from __future__ import print_function
import os
import numpy as np
np.random.seed(33)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

BASE_DIR = '../data'
GLOVE_DIR = BASE_DIR + '/glove.6B/'
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroups/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 50
VALIDATION_SPLIT = 0.2


# 将词和词向量对应起来
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'), encoding="UTF-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')     # asarray和引用功能类似
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# 读取数据文件并对应标签
texts = []  # 文本数据
labels_index = {}  # 存储标签
labels = []  # 存储y
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    path = os.path.join(TEXT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                f = open(fpath, encoding='latin-1')
                texts.append(f.read())
                f.close()
                labels.append(label_id)   # labels和texts是一一对应的

print('Found %s texts.' % len(texts))

# 将样本转化为tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# 割分训练集和测试集
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

score_train = []
score_val = []

start = 0
step = int(VALIDATION_SPLIT * data.shape[0])
end = step
# 五折交叉验证
for i_out in range(0, int(1/VALIDATION_SPLIT)):
    x_train = np.vstack((data[0: start], data[end:]))     # 将两个矩阵纵向连接起来，即每行的维度不变，但是行数增加
    y_train = np.vstack((labels[0: start], labels[end:]))
    x_val = data[start: end]
    y_val = labels[start: end]

    print('第%s轮次： Preparing embedding matrix.' % i_out)

    # 词向量
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # 加载词向量
    embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                input_length=MAX_SEQUENCE_LENGTH,
                                weights=[embedding_matrix],
                                trainable=True)

    print('第%s轮次： Training model.' % i_out)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')    # 转化
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(filters=128, kernel_size=5, activation='tanh')(embedded_sequences)    # (996, 128)
    x = MaxPooling1D(4)(x)     # (249, 128)
    x = Conv1D(128, 5, activation='tanh')(x)    # (245, 128)
    x = MaxPooling1D(5)(x)    # (49, 128)
    x = Conv1D(128, 6, activation='tanh')(x)     # (44, 128)
    x = MaxPooling1D(2)(x)   # (22, 128)
    x = Conv1D(128, 4, activation='tanh')(x)  # (19, 128)
    x = MaxPooling1D(19)(x)  # (1, 128)
    x = Flatten()(x)
    x = Dense(128, activation='tanh')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='Adadelta',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, nb_epoch=5)

    score = model.evaluate(x_train, y_train, verbose=0)
    score_train.append(score)
    print('第%s轮次： train score: %s' % (i_out, score[0]))
    print('第%s轮次： train accuracy: %s' % (i_out, score[1]))
    score = model.evaluate(x_val, y_val, verbose=0)
    score_val.append(score)
    print('第%s轮次： Test score: %s' % (i_out, score[0]))
    print('第%s轮次： Test accuracy: %s' % (i_out, score[1]))

    start += step
    end += step

score_train_mean = np.mean(score_train, axis=0)
score_val_mean = np.mean(score_val, axis=0)

print('平均值： train score: %s' % score_train_mean[0])
print('平均值： train accuracy: %s' % score_train_mean[1])
print('平均值： Test score: %s' % score_val_mean[0])
print('平均值： Test accuracy: %s' % score_val_mean[1])
