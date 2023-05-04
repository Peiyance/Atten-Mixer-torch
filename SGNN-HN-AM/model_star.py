#!/usr/bin/env python36
# -*- coding: utf-8 -*-
import pickle
import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from tqdm import tqdm

class StarGNN(Module):
    def __init__(self, hidden_size, step=1):
        super(StarGNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.input_size = hidden_size * 2
        self.gate_size = 3 * hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.Wq1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wk1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wq2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wk2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.Wg = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def GNNCell(self, A, hidden):

        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A, hidden, mask):

        s = torch.sum(hidden, dim=1) / (torch.sum(mask, dim=1) + 1e-6).unsqueeze(-1)

        s = s.unsqueeze(1)
        hidden0 = hidden
        for i in range(self.step):
            hidden1 = self.GNNCell(A, hidden)

            alpha = self.Wq1(hidden1).bmm(self.Wk1(s).permute(0, 2, 1)) / math.sqrt(self.hidden_size)
            alpha = torch.softmax(alpha, dim=1)
            assert not torch.isnan(s).any()
            assert not torch.isnan(alpha).any()
            assert not torch.isnan(hidden1).any()

            hidden = (1 - alpha) * hidden1 + alpha * s.repeat(1, alpha.size(1), 1)
            assert not torch.isnan(hidden).any()
            beta = self.Wq2(hidden).bmm(self.Wk2(s).permute(0, 2, 1)) / math.sqrt(self.hidden_size)
            mask = mask[:, :beta.size(1)]
            beta = torch.softmax(beta, dim=1)
            assert not torch.isnan(beta).any()
            s = torch.sum(beta * hidden, dim=1, keepdim=True)

        g = self.Wg(torch.cat((hidden0, hidden), dim=-1)).sigmoid_()
        assert not torch.isnan(hidden).any()
        hidden = g * hidden0 + (1 - g) * hidden

        return hidden, s

class LastAttenion(Module):
    def __init__(self, hidden_size, heads, dot, l_p, last_k=3, use_attn_conv=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.heads = heads
        self.last_k = last_k
        self.linear_zero = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.heads, bias=False)
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.linear_five = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.dropout = 0.1
        self.dot = dot
        self.l_p = l_p
        self.use_attn_conv = use_attn_conv
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1)

    def forward(self, ht1, hidden, mask):
        q0 = self.linear_zero(ht1).view(-1, ht1.size(1), self.hidden_size // self.heads)
        q1 = self.linear_one(hidden).view(-1, hidden.size(1), self.hidden_size // self.heads)
        q2 = self.linear_two(hidden).view(-1, hidden.size(1), self.hidden_size // self.heads)
        assert not torch.isnan(q0).any()
        assert not torch.isnan(q1).any()
        alpha = torch.sigmoid(torch.matmul(q0, q1.permute(0, 2, 1)))
        assert not torch.isnan(alpha).any()
        alpha = alpha.view(-1, q0.size(1) * self.heads, hidden.size(1)).permute(0, 2, 1)
        alpha = torch.softmax(2 * alpha, dim=1)
        assert not torch.isnan(alpha).any()
        if self.use_attn_conv == "True":
            m = torch.nn.LPPool1d(self.l_p, self.last_k, stride=self.last_k)
            alpha = m(alpha)
            alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))
            alpha = torch.softmax(2 * alpha, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        a = torch.sum(
            (alpha.unsqueeze(-1) * q2.view(hidden.size(0), -1, self.heads, self.hidden_size // self.heads)).view(
                hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(), 1)
        return a, alpha

class StarSessionGraph(Module):
    def __init__(self, opt, n_node):
        super(StarSessionGraph, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.num_heads = opt.heads
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.gnn = StarGNN(self.hidden_size, step=opt.step)
        self.attn = nn.MultiheadAttention(self.hidden_size, 1)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.linear_four = nn.Linear(self.hidden_size, self.hidden_size)
        self.layernorm1 = nn.LayerNorm(self.hidden_size)
        self.layernorm2 = nn.LayerNorm(self.hidden_size)
        self.layernorm3 = nn.LayerNorm(self.hidden_size)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
        self.norm = opt.norm

        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.last_k = opt.last_k
        self.l_p = opt.l_p
        self.use_attn_conv = opt.use_attn_conv
        self.scale = opt.scale
        self.heads = opt.heads
        self.dot = opt.dot
        self.linear_q = nn.ModuleList()
        for i in range(self.last_k):
            self.linear_q.append(nn.Linear((i + 1) * self.hidden_size, self.hidden_size))
        self.mattn = LastAttenion(self.hidden_size, self.heads, self.dot, self.l_p, last_k=self.last_k,
                                  use_attn_conv=self.use_attn_conv)
        self.gate = nn.Linear(2 * self.hidden_size, 1, bias=False)
        self.dropout = 0.1

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1)

    def compute_scores(self, hidden, s, mask):
        hts = []
        lengths = torch.sum(mask, dim=1)
        for i in range(self.last_k):
            hts.append(self.linear_q[i](torch.cat([hidden[torch.arange(mask.size(0)).long(), torch.clamp(lengths - (j + 1), -1, 1000)] for j in range(i + 1)], dim=-1)).unsqueeze(1))
        ht0 = hidden[torch.arange(mask.size(0)).long(), torch.sum(mask, 1) - 1]
        hts = torch.cat(hts, dim=1)
        hts = hts.div(torch.norm(hts, p=2, dim=1, keepdim=True) + 1e-12)
        hidden = hidden[:, :mask.size(1)]
        ais, weights = self.mattn(hts, hidden, mask)
        a = self.linear_transform(torch.cat((ais.squeeze(), ht0), 1))
        b = self.embedding.weight[1:]
        if self.norm:
            a = a.div(torch.norm(a, p=2, dim=1, keepdim=True) + 1e-12)
            b = b.div(torch.norm(b, p=2, dim=1, keepdim=True) + 1e-12)
        b = F.dropout(b, self.dropout, training=self.training)
        scores = torch.matmul(a, b.transpose(1, 0))
        if self.scale:
            scores = 12 * scores
        return scores, a

    def forward(self, inputs, A, mask):

        hidden = self.embedding(inputs)
        hidden = self.layernorm3(hidden)
        hidden, s = self.gnn(A, hidden, mask)
        return hidden, s

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, i, data):
    alias_inputs, A, items, mask, targets = data.get_slice(i)
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(A).float())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden, s = model(items, A,mask)

    if model.norm:
        seq_shape = list(hidden.size())
        hidden = hidden.view(-1, model.hidden_size)
        norms = torch.norm(hidden, p=2, dim=1)  # l2 norm over session embedding
        hidden = hidden.div(norms.unsqueeze(-1).expand_as(hidden))
        hidden = hidden.view(seq_shape)
    get = lambda i: hidden[i][alias_inputs[i]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])
    if model.norm:
        seq_shape = list(seq_hidden.size())
        seq_hidden = seq_hidden.view(-1, model.hidden_size)
        norms = torch.norm(seq_hidden, p=2, dim=1)  # l2 norm over session embedding
        seq_hidden = seq_hidden.div(norms.unsqueeze(-1).expand_as(seq_hidden))
        seq_hidden = seq_hidden.view(seq_shape)
    sc,a = model.compute_scores(seq_hidden, s, mask)
    return targets, sc, a


def train_test(model, train_data, test_data):
    model.scheduler.step()
    print('start training: ', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i, j in tqdm(zip(slices, np.arange(len(slices))), total=len(slices)):
        model.optimizer.zero_grad()
        targets, scores, a_hidden = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)

    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr, phi = [], [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores, a_hidden = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        phic = 0
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            for s in score:
                # phic += model.count_table[s+1]
                phic += 1
            phic /= 20
            phi.append(phic)
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    phi = np.mean(phi)
    return hit, mrr

def formal_test(model, train_data, test_data):
    print('start predicting: ', datetime.datetime.now())
    model.eval()
    hit, mrr, phi = [], [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores, a_hidden = forward(model, i, test_data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        phic = 0
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            for s in score:
                # phic += model.count_table[s+1]
                phic += 1
            phic /= 20
            phi.append(phic)
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    phi = np.mean(phi)
    return hit, mrr
