##
# 贵州大学 @贾阵
#1196945562@qq.com
##

import numpy as np
import os
class Input_data(object):

    def build_vocab(self,word2vec_path=None):
        if word2vec_path:
            with open(word2vec_path, 'r') as f:
                header = f.readline()
                vocab_size, layer2_size = map(int, header.split())
                init_W = np.random.uniform(-0.25, 0.25, (vocab_size, self.embedding_size))
                print('vocab_size={}'.format(vocab_size))
                dictionary = dict()
                while True:
                    line = f.readline()
                    if not line:
                        break
                    word = line.split()[0]
                    dictionary[word] = len(dictionary)
                    init_W[dictionary[word]] = np.array(line.split()[1:], dtype=np.float32)

            return dictionary, init_W

    def file_to_word2vec_word_ids(self, filename, word_to_id):
        with open(filename, 'r') as f:
            f.readline()  # remove header
            sentences_A = []
            sentencesA_length = []
            sentences_B = []
            sentencesB_length = []
            relatedness_scores = []
            pairIDs = []
            while True:
                line = f.readline()
                if not line: break
                ID = line.split('\t')[0]  # for test
                pairIDs.append(ID)
                sentence_A = line.split('\t')[1]
                sentence_B = line.split('\t')[2]
                relatedness_score = line.split('\t')[3]

                _ = [word_to_id[word] for word in sentence_A.split() if word in word_to_id]
                sentencesA_length.append(len(_))
                _ += [0] * (self.max_length - len(_))
                sentences_A.append(_)

                _ = [word_to_id[word] for word in sentence_B.split() if word in word_to_id]
                sentencesB_length.append(len(_))
                _ += [0] * (self.max_length - len(_))
                sentences_B.append(_)
                relatedness_scores.append((float(relatedness_score)-1)/4)
        assert len(sentences_A) == len(sentencesA_length) == len(sentences_B) == len(sentencesB_length) == len(relatedness_scores)

        np.random.shuffle([sentences_A])
        np.random.shuffle([sentencesA_length])
        np.random.shuffle([sentences_B])
        np.random.shuffle([sentencesB_length])
        np.random.shuffle([relatedness_scores])
        set = [sentences_A, sentencesA_length, sentences_B, sentencesB_length, relatedness_scores]
        return set

    def next_batch(self, start, end, input):
        inputs_A = input[0][start:end]
        inputsA_length = input[1][start:end]
        inputs_B = input[2][start:end]
        inputsB_length = input[3][start:end]
        labels = np.reshape(input[4][start:end], (len(range(start, end)), 1))
        return [inputs_A, inputsA_length, inputs_B, inputsB_length, labels]

    def get_data(self):
        train_path = os.path.join("SICK", 'SICK_all_train.txt')
        test_path = os.path.join("SICK", 'SICK_test_annotated.txt')
        dictionary, init_W = self.build_vocab("embeddings/word2vec_norm.txt")
        train_data = self.file_to_word2vec_word_ids(train_path, dictionary)
        test_data = self.file_to_word2vec_word_ids(test_path, dictionary)
        return train_data, test_data, dictionary, init_W

    def __init__(self, batch_size, embedding_size, max_length):
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.max_length = max_length


