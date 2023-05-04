# /usr/bin/env python36
# -*- coding: utf-8 -*-
import math
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class LastAttenion(Module):

    def __init__(self, hidden_size, heads, dot, l_p, last_k=3, use_lp_pool=False):
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
        self.use_lp_pool = use_lp_pool
        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            weight.data.normal_(std=0.1)

    def forward(self, ht1, hidden, mask):

        q0 = self.linear_zero(ht1).view(-1, ht1.size(1), self.hidden_size // self.heads)
        q1 = self.linear_one(hidden).view(-1, hidden.size(1),
                                          self.hidden_size // self.heads)  # batch_size x seq_length x latent_size
        q2 = self.linear_two(hidden).view(-1, hidden.size(1), self.hidden_size // self.heads)
        assert not torch.isnan(q0).any()
        assert not torch.isnan(q1).any()
        alpha = torch.sigmoid(torch.matmul(q0, q1.permute(0, 2, 1)))
        assert not torch.isnan(alpha).any()
        alpha = alpha.view(-1, q0.size(1) * self.heads, hidden.size(1)).permute(0, 2, 1)
        alpha = torch.softmax(2 * alpha, dim=1)
        assert not torch.isnan(alpha).any()
        if self.use_lp_pool == "True":
            m = torch.nn.LPPool1d(self.l_p, self.last_k, stride=self.last_k)
            alpha = m(alpha)
            alpha = torch.masked_fill(alpha, ~mask.bool().unsqueeze(-1), float('-inf'))
            alpha = torch.softmax(2 * alpha, dim=1)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        a = torch.sum(
            (alpha.unsqueeze(-1) * q2.view(hidden.size(0), -1, self.heads, self.hidden_size // self.heads)).view(
                hidden.size(0), -1, self.hidden_size) * mask.view(mask.shape[0], -1, 1).float(), 1)
        a = self.last_layernorm(a)
        return a, alpha


class SessionGraphAttn(Module):
    def __init__(self, opt, n_node):
        super(SessionGraphAttn, self).__init__()
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.norm = opt.norm
        self.scale = opt.scale
        self.batch_size = opt.batchSize
        self.heads = opt.heads
        self.use_lp_pool = opt.use_lp_pool
        self.softmax = opt.softmax
        self.dropout = opt.dropout
        self.last_k = opt.last_k
        self.dot = opt.dot
        self.embedding = nn.Embedding(self.n_node, self.hidden_size)
        self.l_p = opt.l_p
        self.mattn = LastAttenion(self.hidden_size, self.heads, self.dot, self.l_p, last_k=self.last_k,
                                  use_lp_pool=self.use_lp_pool)
        self.linear_q = nn.ModuleList()
        for i in range(self.last_k):
            self.linear_q.append(nn.Linear((i + 1) * self.hidden_size, self.hidden_size))

        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get(self, i, hidden, alias_inputs):
        return hidden[i][alias_inputs[i]]

    def compute_scores(self, hidden, mask):

        hts = []

        lengths = torch.sum(mask, dim=1)

        for i in range(self.last_k):
            hts.append(self.linear_q[i](torch.cat(
                [hidden[torch.arange(mask.size(0)).long(), torch.clamp(lengths - (j + 1), -1, 1000)] for j in
                 range(i + 1)], dim=-1)).unsqueeze(1))

        ht0 = hidden[torch.arange(mask.size(0)).long(), torch.sum(mask, 1) - 1]

        hts = torch.cat(hts, dim=1)
        hts = hts.div(torch.norm(hts, p=2, dim=1, keepdim=True) + 1e-12)

        hidden1 = hidden
        hidden = hidden1[:, :mask.size(1)]

        ais, weights = self.mattn(hts, hidden, mask)
        a = self.linear_transform(torch.cat((ais.squeeze(), ht0), 1))

        b = self.embedding.weight[1:]

        if self.norm:
            a = a.div(torch.norm(a, p=2, dim=1, keepdim=True) + 1e-12)
            b = b.div(torch.norm(b, p=2, dim=1, keepdim=True) + 1e-12)
        b = F.dropout(b, self.dropout, training=self.training)
        scores = torch.matmul(a, b.transpose(1, 0))
        if self.scale:
            scores = 16 * scores
        return scores

    def forward(self, inputs):

        hidden = self.embedding(inputs)

        if self.norm:
            hidden = hidden.div(torch.norm(hidden, p=2, dim=-1, keepdim=True) + 1e-12)

        hidden = F.dropout(hidden, self.dropout, training=self.training)

        return hidden
