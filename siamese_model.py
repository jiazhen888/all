##
# 贵州大学 @贾阵
#1196945562@qq.com
##

import tensorflow as tf

class Siamesenetwork(object):
    def build_model(self, input_, input_length, hidden_units, num_layer):
        rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units)
        rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=self.dropout_keep_pro)
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * num_layer)
        outputs, last_states = tf.nn.dynamic_rnn(
            cell=rnn_cell,
            dtype=tf.float32,
            sequence_length=input_length,
            inputs=input_
        )
        return outputs, last_states

    def __init__(self, max_length, embedding_size, hidden_units, num_layer, dictionary):

        initializer = tf.contrib.layers.xavier_initializer()
        with tf.variable_scope('Model', initializer=initializer):
            self.sentences_A = tf.placeholder(tf.int32, shape=([None, max_length]), name='sentences_A')
            self.sentencesA_length = tf.placeholder(tf.int32, shape=([None]), name='sentencesA_length')
            self.sentences_B = tf.placeholder(tf.int32, shape=([None, max_length]), name='sentences_B')
            self.sentencesB_length = tf.placeholder(tf.int32, shape=([None]), name='sentencesB_length')
            self.labels = tf.placeholder(tf.float32, shape=([None, 1]), name='relatedness_score_label')
            self.dropout_keep_pro = tf.placeholder(tf.float32)
            self.embedding_placeholder = tf.placeholder(tf.float32, [len(dictionary), embedding_size])

            W = tf.Variable(tf.constant(0.0, shape=[len(dictionary), embedding_size]), trainable=False,name='W')

            self.embedding_init = W.assign(self.embedding_placeholder)

            self.sentences_A_emb = tf.nn.embedding_lookup(params=self.embedding_init, ids=self.sentences_A)
            self.sentences_B_emb = tf.nn.embedding_lookup(params=self.embedding_init, ids=self.sentences_B)


            # model
            with tf.variable_scope('siamese') as scope:
                outputs_A, last_states_A = self.build_model(self.sentences_A_emb, self.sentencesA_length, hidden_units, num_layer)
                scope.reuse_variables()     #参数共享
                output = self.build_model(self.sentences_B_emb, self.sentencesB_length, hidden_units, num_layer)

            logits = tf.concat([last_states_A[num_layer- 1][1], output[1][num_layer - 1][1]], axis=-1)
            self.prediction = tf.layers.dense(logits, 1,
                                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))

            # cost
            self.cost = tf.reduce_mean(tf.square(tf.subtract(self.prediction, self.labels)))    #训练使用
            self.cost2 = tf.reduce_mean(tf.square(tf.subtract(self.prediction * 4 + 1, self.labels * 4 + 1)))  #测试使用