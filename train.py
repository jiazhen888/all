##
# 贵州大学 @贾阵
#1196945562@qq.com
##

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from  input_data import Input_data
from siamese_model import Siamesenetwork
import tensorflow as tf
import numpy as np

flags = tf.flags
flags.DEFINE_string('word2vec_norm','embeddings/word2vec_norm.txt','Word2vec file with pre-trained embeddings')
flags.DEFINE_string('data_path','SICK','SICK data set path')
flags.DEFINE_string('save_path','SICK/STS_log/','STS model output directory')
flags.DEFINE_integer('embedding_size',300,'Dimensionality of word embedding')
flags.DEFINE_integer('max_length',26 ,'one sentence max length words which is in dictionary')
flags.DEFINE_integer('hidden_units',50,'')
flags.DEFINE_bool('use_fp64',False,'Train using 64-bit floats instead of 32bit floats')
FLAGS = flags.FLAGS
class Config(object):
    init_scale = 0.2
    learning_rate =.01
    max_grad_norm = 1.
    keep_prob = 1.
    lr_decay = 0.98
    batch_size = 30
    max_epoch = 22
    max_max_epoch = 300
    num_layer = 1

config=Config()
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

inp = Input_data(Config.batch_size, FLAGS.embedding_size, FLAGS.max_length)
train_data, test_data, dictionary, init_W = inp.get_data()


with tf.Graph().as_default():
    siamese = Siamesenetwork(FLAGS.max_length, FLAGS.embedding_size, FLAGS.hidden_units, Config.num_layer, dictionary)

    learn_rate = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
    gradients, _ = tf.clip_by_global_norm(tf.gradients(siamese.cost, tvars), Config.max_grad_norm)
    train_op = optimizer.apply_gradients(zip(gradients, tvars), global_step=tf.contrib.framework.get_or_create_global_step())
    new_learn_rate = tf.placeholder(tf.float32, shape=[], name='new_learning_rate')
    learn_rate_update = tf.assign(learn_rate, new_learn_rate)

    with tf.Session(config=config_gpu) as sess:
        sess.run(tf.global_variables_initializer())
        # train
        total_batch = int(len(train_data[0]) / config.batch_size)
        prev_train_cost = 1
        for epoch in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(epoch + 1 - config.max_epoch, 0.0)
            sess.run([learn_rate, learn_rate_update], feed_dict={new_learn_rate: config.learning_rate * lr_decay})
            print('Epoch {} Learning rate: {}'.format(epoch, sess.run(learn_rate)))

            avg_cost = 0.
            for i in range(total_batch):
                start = i * config.batch_size
                end = (i + 1) * config.batch_size

                next_batch_input = inp.next_batch(start, end, train_data)
                _, train_cost, train_predict = sess.run([train_op, siamese.cost, siamese.prediction], feed_dict={
                    siamese.sentences_A: next_batch_input[0],
                    siamese.sentencesA_length: next_batch_input[1],
                    siamese.sentences_B: next_batch_input[2],
                    siamese.sentencesB_length: next_batch_input[3],
                    siamese.labels: next_batch_input[4],
                    siamese.dropout_keep_pro: config.keep_prob,
                    siamese.embedding_placeholder: init_W
                })
                avg_cost += train_cost

            start = total_batch * config.batch_size
            end = len(train_data[0])
            if not start == end:
                next_batch_input = inp.next_batch(start, end, train_data)
                _, train_cost, train_predict = sess.run([train_op, siamese.cost, siamese.prediction], feed_dict={
                    siamese.sentences_A: next_batch_input[0],
                    siamese.sentencesA_length: next_batch_input[1],
                    siamese.sentences_B: next_batch_input[2],
                    siamese.sentencesB_length: next_batch_input[3],
                    siamese.labels: next_batch_input[4],
                    siamese.dropout_keep_pro: config.keep_prob,
                    siamese.embedding_placeholder: init_W
                })
                avg_cost += train_cost

            if prev_train_cost > avg_cost / total_batch:
                print('Average cost:\t{} ↓'.format(avg_cost / total_batch))
            else:
                print('Average cost:\t{} ↑'.format(avg_cost / total_batch))
            prev_train_cost = avg_cost / total_batch

        test_cost, test_predict = sess.run([siamese.cost, siamese.prediction], feed_dict={
            siamese.sentences_A: test_data[0],
            siamese.sentencesA_length: test_data[1],
            siamese.sentences_B: test_data[2],
            siamese.sentencesB_length: test_data[3],
            siamese.labels: np.reshape(test_data[4], (len(test_data[4]), 1)),
            siamese.embedding_placeholder: init_W,
            siamese.dropout_keep_pro: 1.0
        })
        print(test_cost)


