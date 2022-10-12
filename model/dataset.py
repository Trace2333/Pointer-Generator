import pickle
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DatasetBase(Dataset):
    """包装数据，需要包装最后的padding mask与OOV词汇"""
    def __init__(self, input_ids_path, y_path, vocab_path, oov_words_path):
        super(DatasetBase, self).__init__()
        with open(input_ids_path, "rb") as f1:
            self.ids = pickle.load(f1)
        with open(y_path, "rb") as f2:
            y = pickle.load(f2)
            self.y = y
        with open(vocab_path, "rb") as f3:
            vocab = pickle.load(f3)
            self.vocab = vocab
        with open(oov_words_path, "rb") as f4:
            oov_words = pickle.load(f4)
            self.oov_words = oov_words

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        """
            固定取前三句作为整个article的摘要。暂定为三句连成一句来预测
        Return:
            x:input_ids，oov words list, oov
            y:y_ids（y中也有x中的oov词，但是需要在x中的列表做查询来还原）, num_oov_words
            Note：
                没有对attention和padding做mask，所以训练速度和loss将会一定程度上增大
        """
        abstracts = self.y[item]
        articles = self.ids[item]
        oov_words = self.oov_words[item]
        if len(abstracts) > 3:
            abstracts = abstracts[:3]
        max_y_length = len(oov_words)
        return (articles, oov_words), (abstracts, max_y_length)


def collate_fn(batch):
    """
    自定义collate function，用来处理输入并整合
    :param batch: list of elem like:((articles, oov_words), (abstracts, max_y_length))
    :return:
    """
    articles = [back_tuple[0][0] for back_tuple in batch]   # NO PAD
    oov_words = [back_tuple[0][1] for back_tuple in batch]
    abstracts = [back_tuple[1][0] for back_tuple in batch]
    max_oov_nums = [back_tuple[1][1] for back_tuple in batch]
    return (articles, oov_words, abstracts, max_oov_nums)


if __name__ == '__main__':
    """For Test"""
    dataset = DatasetBase(
        input_ids_path="../dataset/ids.pkl",
        y_path="../dataset/y.pkl",
        vocab_path="../dataset/word_id.pkl",
        oov_words_path="../dataset/oov_words.pkl",
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    for i in loader:
        print(i)