import torch
import torch.nn as nn
from functools  import reduce
import pickle
a = torch.randn([8, 20, 30])
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
print(out)




