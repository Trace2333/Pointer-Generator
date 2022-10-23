import torch
import torch.nn as nn
from functools import reduce
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.fc_state = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.W_h = nn.Linear(2 * hidden_size, 2 * hidden_size, bias=False)
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.init_weight()

    def encode(self, x, seq_lens):
        # 打包
        packed_x = pack_padded_sequence(x, seq_lens.cpu(), batch_first=True, enforce_sorted=False)

        (outputs, (final_hidden_state, cell_state)) = self.lstm(packed_x)
        # 解包
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        forward_final_state, backward_final_state = final_hidden_state
        forward_cell_state, backward_cell_state = cell_state

        return outputs, forward_final_state, backward_final_state, forward_cell_state, backward_cell_state

    def state_cell_reproduce(self, forward_final_state, backward_final_state, forward_cell_state, backward_cell_state):
        cell_state = torch.concat((forward_cell_state, backward_cell_state), dim=1)
        final_state = torch.concat((forward_final_state, backward_final_state), dim=1)
        re_cell_state = self.fc_cell(cell_state)
        re_final_state = self.fc_state(final_state)
        return re_cell_state, re_final_state

    def forward(self, x, seq_lens):
        """x是已经转换完的词嵌入矩阵"""
        outputs, forward_final_state, backward_final_state, forward_cell_state, backward_cell_state = self.encode(x, seq_lens)
        cell_state, state = self.state_cell_reproduce(forward_final_state, backward_final_state, forward_cell_state, backward_cell_state)
        return outputs, cell_state, state

    def init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

    def init_zero_state(self):
        return torch.zeros([2, self.batch_size, self.hidden_size])

    def init_random_state(self):
        return torch.randn([2, self.batch_size, self.hidden_size])


class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, input_size, embw_size, attention_size, batch_size, device, if_coverage=True):
        #   目前的一些问题：
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=False, batch_first=True)
        self.embw = nn.Embedding(vocab_size, embw_size)   # When no input Embedding vec
        self.W_s = nn.Linear(hidden_size * 2, attention_size, bias=True)
        self.W_h = nn.Linear(hidden_size * 2, attention_size)
        self.w_c = nn.Linear(1, attention_size, bias=False)
        self.v = nn.Linear(attention_size, 1, bias=False)
        self.decode_fc1 = nn.Linear(hidden_size * 3, hidden_size * 2, bias=True)
        self.decode_fc2 = nn.Linear(hidden_size * 2, vocab_size, bias=True)   # 线性层放大可能损失信息
        self.p_gen_fc = nn.Linear(embw_size + attention_size * 2, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.atten_size = attention_size
        self.atten_vec_size = hidden_size   # 认为是attention size
        self.last_cell_state = None
        self.last_hidden_state = None
        self.if_coverage = if_coverage
        self.batch_size = batch_size
        #self.attention_dists = []
        self.coverages = None
        #self.cov_loss = []
        #self.gen_prob = []
        self.device = device
        self.exteneded_size = 0
        self.weight_init()

    def lstm_compute(self, x):
        if self.last_hidden_state is None:   # 初始化一个零矩阵
            self.last_hidden_state = torch.zeros([1, self.batch_size, self.hidden_size]).to(self.device)

        if self.last_cell_state is None:
            self.last_cell_state = torch.zeros([1, self.batch_size, self.hidden_size]).to(self.device)

        (out, (decoder_state, decoder_cell)) = self.lstm(x, (self.last_hidden_state, self.last_cell_state))
        self.last_hidden_state = decoder_state.data
        self.last_cell_state = decoder_cell.data

        decoder_cell = decoder_cell.permute(1, 0, 2)
        decoder_state = decoder_state.permute(1, 0, 2)
        return decoder_state, decoder_cell

    def attention_dist_get(self, encoder_hidden_state, decoder_state):
        """当前时间步的decoder out"""
        # 每一个时间步都会产生一个attention distribution,最后的coverage需要将所有的attention串起来
        # decoder state需要连接cell和state
        b, l, n = list(encoder_hidden_state.size())

        if self.coverages is None or self.coverages.size()[1] != l:
            self.coverages = torch.zeros([b, l]).to(self.device)

        encoder_features = self.W_h(encoder_hidden_state).view(-1, n)
        decoder_features = self.W_s(decoder_state).view(-1, n)
        decoder_features = decoder_features.unsqueeze(1).expand(b, l, self.atten_size).reshape(-1, n)   # 用contiguous节省内存

        if self.if_coverage is True:   # 初始化coverage
            coverage_input = self.coverages.view(-1, 1)
            coverage_features = self.w_c(coverage_input)
            e_t = self.tanh(encoder_features + decoder_features + coverage_features)
        else:   # 不用coverage
            e_t = self.tanh(encoder_features + decoder_features)

        re_e_t = self.v(e_t).view(-1, l)   # 重整计算一次
        attention = self.softmax(re_e_t).unsqueeze(1)
        #self.attention_dists.append(attention)
        self.coverages = self.coverages + attention.squeeze(1).data

        return attention   # [batch_size, atten_length]

    def context_vec_get(self, attention, encoder_outputs):
        context_vec = torch.bmm(attention.unsqueeze(1), encoder_outputs).view(-1, self.hidden_size * 2)
        return context_vec   # weighted sum of the encoder_outputs, [batch_size, hidden_size * 2]

    def vocab_dist_get(self, context_vec, decoder_state):
        """当前时间步"""
        decode_input = torch.cat((context_vec, decoder_state), dim=-1)
        fc1_out = self.decode_fc1(decode_input)
        fc2_out = self.decode_fc2(fc1_out)   # 好像应该是一个线性层
        vocab_dist = self.softmax(fc2_out)
        return vocab_dist

    def coverage_mechanism(self):
        #   简单将过往的各层attention_dist加和即可
        return reduce(lambda x, y: x+y, self.attention_dists)

    def cov_loss(self):
        #   attention_dists 和 coverage都是列表形式，元素为tensor
        for atten_dis, cov in zip(self.attention_dists, self.coverages):
            min_one = torch.minimum(atten_dis, cov)
            cov_loss = torch.sum(min_one, dim=-1)
            #self.cov_loss.append(cov_loss)

    def gen_prob_get(self, context_vec, decoder_input, decoder_state_cell):
        """实现正确性存疑，可能是三个张量连接起来然后做一次全连接变换"""
        gen = torch.cat((context_vec, decoder_state_cell, decoder_input), 1)
        p_gen = self.p_gen_fc(gen)
        p_gen = torch.sigmoid(p_gen)

        #self.gen_prob.append(p_gen)

        return p_gen   # 直接返回的是当前时间步

    def forward(self, x, y, encoder_in, max_oov_nums, ids):
        """y for teacher forcing, encoder_in->[encoder_outputs, encoder_cell, encoder_state]
        ,x is embedding.外包装循环"""
        decoder_state_t, decoder_cell_t = self.lstm_compute(y)
        decoder_state_cell = torch.cat((decoder_state_t, decoder_cell_t), dim=-1)

        attention_dist = self.attention_dist_get(encoder_in[0], decoder_state_cell.squeeze(1)).squeeze(1)

        context_vec = self.context_vec_get(attention_dist, encoder_in[0])

        p_vocab = self.vocab_dist_get(context_vec, decoder_state_t.squeeze(1))

        gen_prob = self.gen_prob_get(context_vec, x.squeeze(1), decoder_state_cell.squeeze(1))

        # 概率做乘积
        attention_dist = (1 - gen_prob) * attention_dist
        p_vocab = gen_prob * p_vocab

        #连接额外的0
        extended_zeros = torch.zeros([self.batch_size, max_oov_nums]).to(self.device)
        p_vocab = torch.cat((p_vocab, extended_zeros), -1)

        p = p_vocab.scatter_add(1, ids, attention_dist)
        return p

    def init_zero_state(self):
        return torch.zeros([2, self.batch_size, self.hidden_size])

    def init_random_state(self):
        return torch.randn([2, self.batch_size, self.hidden_size])

    def weight_init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=10)


class PointerGenerator(nn.Module):
    """每个batch的输入包含了需要的所有信息"""
    def __init__(self, if_pointer, if_coverage, input_size, hidden_size, vocab_size, batch_size, embw_size, attention_size, device):
        super(PointerGenerator, self).__init__()
        self.encoder = Encoder(input_size=input_size, hidden_size=hidden_size, batch_size=batch_size)
        self.decoder = Decoder(vocab_size=vocab_size,
                               hidden_size=hidden_size,
                               input_size=input_size,
                               embw_size=embw_size,
                               attention_size=attention_size,
                               batch_size=batch_size,
                               if_coverage=if_coverage,
                               device=device)
        self.batch_size = batch_size
        self.device = device
        self.vocab_size = vocab_size
        self.embw = nn.Embedding(vocab_size, embw_size)   # 随机生成词矩阵

    def forward(self, x, oov_words, y, max_oov_nums, seq_lens):   # y for teacher-forcing
        encoder_input = self.embw(x)
        encoder_input_y = self.embw(y)

        encoder_outputs, encoder_cell_state, encoder_hidden_state = self.encoder.forward(encoder_input, seq_lens)
        encoder_in = [encoder_outputs, encoder_cell_state, encoder_hidden_state]

        out = torch.zeros([self.batch_size, 1, self.vocab_size + max_oov_nums]).to(self.device)

        for token, y_token in zip(encoder_input.split(1, dim=1), encoder_input_y.split(1, dim=1)):
            p = self.decoder.forward(token, y_token, encoder_in, max_oov_nums, x).unsqueeze(1)
            out = torch.cat((out, p), dim=1)
        return out

        # encoder的所有输入都集合到一个时间步中


