#import tensorflow as tf
import struct
import json
import string
import pickle
import torch
import glob
#from tensorflow.core.example import example_pb2
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence, pad_packed_sequence
"""a = torch.randn([8, 20, 30])
b = torch.cat((a[0], a[1]), dim=-1)
print(b.size())

lstm = nn.LSTM(30, 40, bidirectional=True)
(out, (cell, f_state)) = lstm(a)
#out = torch.cat((out[0], out[1]), dim=-1)
print("out:", out.size())
print(f_state.size())
print("cell size:", cell.size())


#    W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
#    encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME") # shape (batch_size,attn_length,1,attention_vec_size)
encoder_state = out
atten_size = 40 * 2
atten_len = 16
atten_vec = 50
encoder_state = encoder_state.unsqueeze(2)
print("encoder_state:", encoder_state.size())
wh = torch.randn([1, 1, atten_size, atten_vec])
#con = nn.Conv2d(in_channels=16, out_channels=atten_vec, kernel_size=(1, 1))
#coned = con(encoder_state)
#print("conv2d:", coned.size())
fc = nn.Linear(in_features=atten_size, out_features=atten_vec, bias=False)
en_f = fc(encoder_state)
print(en_f.size())


ss1 = torch.randn([4, 20, 30]).split(1, dim=1)
ss2 = torch.randn([4, 20]).split(1, dim=1)
#print((ss1[0].size()))
#print(ss2[0].squeeze(1))
#print((ss2[0] * ss1[0]).size())
context_vec = torch.tensor([(i.squeeze(1) * j).tolist() for i, j in zip(ss1, ss2)]).permute(1, 0, 2)
sum = torch.sum(context_vec, dim=1)
print(context_vec.size())
print(sum.size())


list1 = [torch.randn([2, 10]), torch.randn([2, 10]), torch.randn([2, 10])]
out1 = reduce(lambda x, y: x+y, list1)
print(out1)

t1 = torch.randn([2, 10])
t2 = torch.randn([2, 10])
t3 = torch.minimum(t1, t2)
print(t3.size())

input_t = torch.randn([16, 30])
cell = torch.randn([1, 1, 40])   # [batch_size, hidden_size]
state = torch.randn([1, 1, 40])   # [batch_size, hidden_size]
lstm = nn.LSTM(30, 40)
out = lstm(input_t, (state, cell))
print(out)"""
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
        json.dump(json_data, f2)
        print(target_filename, "saved !")


#example_to_json("../dataset/train.bin", "../dataset/train.json")
#example_to_json("../dataset/test.bin", "../dataset/test.json")
#example_to_json("../dataset/val.bin", "../dataset/val.json")
example_to_json("../dataset/chunked/train_000.bin", "../dataset/train000.json")
example_to_json("../dataset/chunked/train_001.bin", "../dataset/train001.json")
example_to_json("../dataset/chunked/train_002.bin", "../dataset/train002.json")
example_to_json("../dataset/chunked/train_003.bin", "../dataset/train003.json")"""
""""with open() as f:
    vocab = pickle.load(f)
print(vocab)

with open("../dataset/train000.json") as f:
    train = json.load(f)
flag = 0
y = []
for i in train:
    tokens = train[i]['abstract'].split('<s>')
    y.append(tokens[1].split())
    y.append(tokens[2].split())
    y.append(tokens[3].split())
    y.append(tokens[4].split())
    for sen in y:
        sen.insert(0, '<s>')
    print(y)"""""
"""atten = torch.randn([2, 30])  # [batch_size, atten_len]
vocab_dist = torch.randn([2, 10004])   # [batch_size, extended_vocab_size]
vocab = torch.tensor(range(10004))
dis = torch.tensor([1, 0, 24, 234, 23, 14]  # 这个表示一句话的输入,每个标号表示那个时间步的
                    )
vocab_dist = vocab_dist.scatter_add(1, dis, atten)
print(vocab_dist.size())

with open("../dataset/oov_words.pkl", "rb") as f:
    oov = pickle.load(f)
#print(oov)

out = torch.randn([4, 10, 300])
tag = torch.ones([4, 10], dtype=torch.int64)
print(out.size())
s = torch.gather(out, 1, tag.unsqueeze(1)).squeeze()
print(s.size())
soft = torch.nn.Softmax(dim=-1)
s = soft(s)
loss = -torch.log(s + 1e-12)
loss = torch.sum(loss) / 300
print(loss)
"""


from rouge import Rouge
import string
pred = torch.randint(0, 10, (16, 20)).tolist()
test = torch.randint(0, 10, (16, 20)).tolist()
f1s = []
f2s = []
for sen in pred:
    f1 = ""
    for i in sen:
        f1 += str(i)+ " "
    f1s.append(f1)
for sen in test:
    f2 = ""
    for i in sen:
        f2 += str(i)+ " "
    f2s.append(f2)
rouge = Rouge()
scores = rouge.get_scores(f1s, f2s, avg=True)
r1_f = scores["rouge-1"]["f"]
r2_f = scores["rouge-2"]["f"]
rl_f = scores["rouge-l"]["f"]
