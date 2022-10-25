import os
import pickle
import json
import struct

from tqdm import tqdm

from tensorflow.core.example import example_pb2

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
    def __init__(self, if_chucked, vocab_file, num_chucked=None):
        """
        用来做综合数据处理，将数据从tensorflow Example处理为ids并存储
        :param if_chucked: True or False, chucked的时候自动读内置的文件表而不是外部输入，
            不使用chucked数据的时候需要指定输入的chucked数量
        :param vocab_file: vocab文件路径
        """
        self.bin_files = [
            "../dataset/train.bin",
            "../dataset/val.bin",
            "../dataset/test.bin",
        ]

        self.chucked_bin_files = os.listdir("../dataset/chunked")
        chucked_bin_files = []
        for i in range(len(self.chucked_bin_files)):
            if "bin" in self.chucked_bin_files[i]:
                chucked_bin_files.append(self.chucked_bin_files[i])
        self.chucked_bin_files = chucked_bin_files

        for i in range(len(self.chucked_bin_files)):
            self.chucked_bin_files[i] = "../dataset/chunked/" + self.chucked_bin_files[i]

        self.chunked_train_bin_files = []
        self.chunked_test_bin_files = []
        self.chunked_val_bin_files = []
        for i in self.chucked_bin_files:
            if "train" in i:
                self.chunked_train_bin_files.append(i)
            if "test" in i:
                self.chunked_test_bin_files.append(i)
            if "val" in i:
                self.chunked_val_bin_files.append(i)

        self.if_chucked = if_chucked
        self.num_chucked = num_chucked
        self.if_bin_to_json = False

        if self.if_chucked is False and self.if_bin_to_json is False:

            for bin_file in self.bin_files:
                if ".bin" in bin_file and os.path.exists(bin_file):
                    self.example_to_json(bin_file, bin_file.replace("bin", "json"))
            self.if_bin_to_json = True

            for i in range(len(self.bin_files)):
                self.bin_files[i] = self.bin_files[i].replace(".bin", ".json")

        if self.if_chucked is True and self.if_bin_to_json is False and self.num_chucked is not None:

            for chunked in [self.chunked_train_bin_files, self.chunked_test_bin_files, self.chunked_val_bin_files]:
                if "train" in chunked[0]:
                    for bin_file in chunked[:self.num_chucked]:
                        if ".bin" in bin_file and os.path.exists(bin_file):
                            self.example_to_json(bin_file, bin_file.replace("bin", "json"))
                    continue

                for bin_file in chunked:
                    if ".bin" in bin_file and os.path.exists(bin_file):
                        self.example_to_json(bin_file, bin_file.replace("bin", "json"))
                self.if_bin_to_json = True

            for chunked in [self.chunked_train_bin_files, self.chunked_test_bin_files, self.chunked_val_bin_files]:
                if "train" in chunked[0]:
                    for i in range(self.num_chucked):
                        self.chunked_train_bin_files[i] = self.chunked_train_bin_files[i].replace(".bin", ".json")
                elif "test" in chunked[0]:
                    for i in range(len(chunked)):
                        self.chunked_test_bin_files[i] = self.chunked_test_bin_files[i].replace(".bin", ".json")
                elif "val" in chunked[0]:
                    for i in range(len(chunked)):
                        self.chunked_val_bin_files[i] = self.chunked_val_bin_files[i].replace(".bin", ".json")

        if self.if_chucked is False:
            self.file_set = []
            for json_file in self.bin_files:
                with open(json_file, "r") as f1:
                    self.file_set.append(json.load(f1))

        if self.if_chucked is True:
            self.file_set = {
                "train": [],
                "test": [],
                "val": [],
            }

            for chunked in (self.chunked_train_bin_files, self.chunked_test_bin_files, self.chunked_val_bin_files):
                if "train" in chunked[0]:
                    for json_file in chunked[:self.num_chucked]:
                        with open(json_file, "r") as f1:
                            self.file_set["train"].append(json.load(f1))

                elif "test" in chunked[0]:
                    for json_file in chunked:
                        with open(json_file, "r") as f1:
                            self.file_set["test"].append(json.load(f1))

                elif "val" in chunked[0]:
                    for json_file in chunked:
                        with open(json_file, "r") as f1:
                            self.file_set["val"].append(json.load(f1))

        with open(vocab_file, 'rb') as f3:
            self.vocab = pickle.load(f3)
        self.vocab_len = len(self.vocab)

    # 需要导入tensorflow,因此预先注释掉
    def example_to_json(self, filename, target_filename):
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
            json.dump(json_data, f2)

    def abstract_extract(self, s=None):
        """提取 ”摘要“
        Args：
            无
        :returns
            提取出的摘要，为嵌套列表形式，已添加头和尾
        """
        ys = []

        if s == "train":
            file_set = self.file_set["train"]
        elif s == "test":
            file_set = self.file_set["test"]
        elif s == "val":
            file_set = self.file_set["val"]
        else:
            file_set = self.file_set

        for json_iter in file_set:
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

    def article_extract(self, s=None):
        """  提取”文本“
        Args:
            无
        :returns
            提取出的文本tokens
        """
        tokens = []
        if s == "train":
            file_set = self.file_set["train"]
        elif s == "test":
            file_set = self.file_set["test"]
        elif s == "val":
            file_set = self.file_set["val"]
        else:
            file_set = self.file_set

        for file in file_set:
            for i in tqdm(file):
                example = file[i]['article'].split()
                tokens.append(example)

        self.tokens = tokens

        return tokens

    def token_to_id(self, s=None):
        """
        将摘要和文本转为ids并存盘
        *为节省内存直接存储对应的id列表为pickle序列文件*
        :return
            完成提取的tokens列表
        """
        ids = []
        oov_words = []

        for story in self.tokens:
            oov_id = self.vocab_len
            id_per_article = [2]
            oov_words_one_story = []
            for token in story:
                if token not in self.vocab:
                    if token in oov_words_one_story:
                        id_per_article.append(oov_words_one_story.index(token) + self.vocab_len)
                    id_per_article.append(oov_id)
                    oov_id += 1
                    if token not in oov_words_one_story:
                        oov_words_one_story.append(token)
                else:
                    id_per_article.append(self.vocab[token])
            id_per_article.append(3)
            ids.append(id_per_article)
            oov_words.append(oov_words_one_story)

        y_ids = []
        for step, abstracts in enumerate(self.ys):   # 常规读取操作
            id_abstracts = []
            for abstract in abstracts[0]:
                id_per_abstract = [2]
                for token in abstract:
                    if token not in self.vocab:
                        if token in oov_words[step]:
                            id_per_abstract.append(self.vocab_len + oov_words[step].index(token))   # 没有在vocab但是在article中的词，记录为在article的num
                        else:
                            id_per_abstract.append(self.vocab['[UNK]'])   # 没有在article也没有在vocab中的词记录为unk
                    else:
                        id_per_abstract.append(self.vocab[token])
                id_per_abstract.append(3)
                id_abstracts.append(id_per_abstract)
            y_ids.append(id_abstracts)

        if s == "train":
            if not os.path.exists("../dataset/train"):
                os.mkdir("../dataset/train/")
            with open("../dataset/train/ids.pkl", "wb") as f1:
                pickle.dump(ids, f1)
                print("../dataset/train/ids.pkl", " Saved !")
            with open("../dataset/train/y.pkl", "wb") as f2:
                pickle.dump(y_ids, f2)
                print("../dataset/train/y_ids.pkl", " Saved !")
            with open("../dataset/train/oov_words.pkl", "wb") as f3:
                pickle.dump(oov_words, f3)
                print("../dataset/train/oov_words.pkl Saved !")
        elif s == "test":
            if not os.path.exists("../dataset/test"):
                os.mkdir("../dataset/test/")
            with open("../dataset/test/ids.pkl", "wb") as f1:
                pickle.dump(ids, f1)
                print("../dataset/test/ids.pkl", " Saved !")
            with open("../dataset/test/y.pkl", "wb") as f2:
                pickle.dump(y_ids, f2)
                print("../dataset/test/y_ids.pkl", " Saved !")
            with open("../dataset/test/oov_words.pkl", "wb") as f3:
                pickle.dump(oov_words, f3)
                print("../dataset/test/oov_words.pkl Saved !")
        elif s == "val":
            if not os.path.exists("../dataset/val"):
                os.mkdir("../dataset/val/")
            with open("../dataset/val/ids.pkl", "wb") as f1:
                pickle.dump(ids, f1)
                print("../dataset/val/ids.pkl", " Saved !")
            with open("../dataset/val/y.pkl", "wb") as f2:
                pickle.dump(y_ids, f2)
                print("../dataset/vak/y_ids.pkl", " Saved !")
            with open("../dataset/val/oov_words.pkl", "wb") as f3:
                pickle.dump(oov_words, f3)
                print("../dataset/val/oov_words.pkl Saved !")
        else:
            with open("../dataset/ids.pkl", "wb") as f1:
                pickle.dump(ids, f1)
                print("../dataset/ids.pkl", " Saved !")
            with open("../dataset/y.pkl", "wb") as f2:
                pickle.dump(y_ids, f2)
                print("../dataset/y_ids.pkl", " Saved !")
            with open("../dataset/oov_words.pkl", "wb") as f3:
                pickle.dump(oov_words, f3)
                print("../dataset/oov_words.pkl Saved !")

        return (ids, y_ids, oov_words)

    def data_process(self):
        """
        数据总处理
        :return:
            无
        """
        self.abstract_extract("train")
        self.article_extract("train")
        self.token_to_id("train")

        self.abstract_extract("test")
        self.article_extract("test")
        self.token_to_id("test")

        self.abstract_extract("val")
        self.article_extract("val")
        self.token_to_id("val")


if __name__ == '__main__':
    vocab_process = Vocab("../dataset/vocab", max_size=50000)
    vocab_process.write_metadata("../dataset/id_word.pkl")
    vocab_process.write_word_to_id("../dataset/word_id.pkl")
    data = Data(
        if_chucked=True,
        vocab_file="../dataset/word_id.pkl",
        num_chucked=10,
    )
    data.data_process()
