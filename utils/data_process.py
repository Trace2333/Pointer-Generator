import os
import pickle
#import struct
import json
#from tensorflow.core.example import example_pb2
# For tensorflow


# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
PAD_TOKEN = '[PAD]'  # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]'  # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]'  # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]'  # This has a vocab id, which is used at the end of untruncated target sequences


class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'rb') as vocab_f:
            for line in vocab_f:
                pieces = line.decode().split()
                if len(pieces) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    break

        print("Finished constructing vocabulary of %i total words. Last word added: %s" % (
            self._count, self._id_to_word[self._count - 1]))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count

    def write_metadata(self, fpath):
        print("Writing word to id file at %s" % fpath)
        with open(fpath, "wb") as f:
            pickle.dump(self._word_to_id, f)

    def write_word_to_id(self, filepath):
        print("Writing word to id file at %s" % filepath)
        with open(filepath, "wb") as f:
            pickle.dump(self._word_to_id, f)


class Data(object):
    def __init__(self, if_chucked, file_or_file_list=None):
        """dataset_path means the dataset dir path"""
        if if_chucked is False:
            with open(file_or_file_list, "r") as f1:
                self.file_set =json.load(f1)
        if if_chucked is True and file_or_file_list is not None:
            self.file_set = []
            for filename in file_or_file_list:
                if os.path.exists(filename):
                    with open(filename, "r") as f1:
                        self.file_set.append(json.load(filename))


    # Need to import tensorflow
    """def example_to_json(filename, target_filename):
        json_data = {}
        count = 0
        with open(filename, "rb") as f1:
            while True:
                count += 1
                per_iter = {}
                len_bytes = f1.read(8)
                if not len_bytes:
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, f1.read(str_len))[0]
                ex = example_pb2.Example.FromString(example_str)
                article = ex.features.feature['article'].bytes_list.value[0].decode()
                abstract = ex.features.feature['abstract'].bytes_list.value[0].decode()
                json_data[count] = per_iter
                per_iter['abstract'] = abstract
                per_iter['article'] = article
                print("Precessed", count, "example")
        with open(target_filename, "w") as f2:
            json.dump(json_data, f2)"""

    def


if __name__ == '__main__':
    vocab_process = Vocab("../dataset/vocab", max_size=80000)
    vocab_process.write_metadata("../dataset/id_word.pkl")
    vocab_process.write_word_to_id("../dataset/word_id.pkl")
   # data_process()