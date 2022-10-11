import os
import pickle
from tqdm import tqdm
import json

#import struct
#from tensorflow.core.example import example_pb2

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
    def __init__(self, if_chucked, vocab_file, file_or_file_list=None):
        """
        用来做综合数据处理，将数据从tensorflow Example处理为ids并存储
        :param if_chucked: True or False
        :param vocab_file: vocab文件路径
        :param file_or_file_list: 待处理的文件列表或文件
        """
        if if_chucked is False:   # 统一分为两种情况做处理
            with open(file_or_file_list, "r") as f1:
                self.file_set =json.load(f1)
        if if_chucked is True and file_or_file_list is not None:
            self.file_set = []
            for filename in file_or_file_list:
                if os.path.exists(filename):
                    with open(filename, "r") as f2:
                        self.file_set.append(json.load(f2))
        self.if_chucked = if_chucked
        with open(vocab_file, 'rb') as f3:
            self.vocab = pickle.load(f3)

    # 需要导入tensorflow,因此预先注释掉
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

    def abstract_extract(self):
        """提取 ”摘要“
        Args：
            无
        :returns
            提取出的摘要，为嵌套列表形式，已添加头和尾
        """
        if self.if_chucked is False:
            ys = []
            for i in tqdm(self.file_set):
                y = []
                example = self.file_set[i]['abstract'].split('<s>')
                y.append(example[1].split())
                y.append(example[2].split())
                y.append(example[3].split())
                y.append(example[4].split())
                for sen in y:
                    sen.insert(0, '<s>')
                ys.append(y)
            self.ys = ys
            return ys
        else:
            ys = []
            for json_iter in self.file_set:
                for i in tqdm(json_iter):
                    y = []
                    example = json_iter[i]['abstract'].split('<s>')
                    for not_split in example[1:]:
                        y.append(not_split.split())
                    for sen in y:
                        sen.insert(0, '<s>')
                    ys.append([y])
            self.ys = ys
            return ys

    def article_extract(self):
        """  提取”文本“
        Args:
            无
        :returns
            提取出的文本tokens
        """
        if self.if_chucked is False:
            tokens = []
            for i in tqdm(self.file_set):
                example = self.file_set[i]['article']
                tokens.append(example)
            self.tokens =tokens
            return tokens
        else:
            tokens = []
            for file in self.file_set:
                for i in tqdm(file):
                    example = file[i]['article'].split()
                    tokens.append(example)
            self.tokens = tokens
            return tokens

    def token_to_id(self):
        """
        将摘要和文本转为ids并存盘
        *为节省内存直接存储对应的id列表为pickle序列文件*
        :return
            完成提取的tokens列表
        """
        y_ids = []
        for abstracts in self.ys:   # 常规读取操作
            id_abstracts = []
            for abstract in abstracts[0]:
                id_per_abstract = [2]
                for token in abstract:
                    if token not in self.vocab:
                        id_per_abstract.append(self.vocab['[UNK]'])
                    else:
                        id_per_abstract.append(self.vocab[token])
                id_per_abstract.append(3)
                id_abstracts.append(id_per_abstract)
            y_ids.append(id_abstracts)
        ids = []
        for abstract in self.tokens:
            id_per_article = [2]
            for token in abstract:
                if token not in self.vocab:
                    id_per_article.append(self.vocab['[UNK]'])
                else:
                    id_per_article.append(self.vocab[token])
            id_per_article.append(3)
            ids.append(id_per_article)
        with open("../dataset/ids.pkl", "wb") as f1:
            pickle.dump(ids, f1)
            print("../dataset/ids.pkl", " Saved!")
        with open("../dataset/y.pkl", "wb") as f2:
            pickle.dump(y_ids, f2)
            print("../dataset/y_ids.pkl", " Saved!")
        return (ids, y_ids)

    def data_process(self):
        """
        数据总处理
        :return:
            无
        """
        self.abstract_extract()
        self.article_extract()
        self.token_to_id()


if __name__ == '__main__':
    vocab_process = Vocab("../dataset/vocab", max_size=80000)
    vocab_process.write_metadata("../dataset/id_word.pkl")
    vocab_process.write_word_to_id("../dataset/word_id.pkl")
    data = Data(
        if_chucked=True,
        vocab_file="../dataset/word_id.pkl",
        file_or_file_list=["../dataset/train000.json",
                           "../dataset/train001.json",
                           "../dataset/train002.json",
                           "../dataset/train003.json"]
    )
    data.data_process()
