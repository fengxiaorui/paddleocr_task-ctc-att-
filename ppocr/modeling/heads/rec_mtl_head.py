# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
from paddle import ParamAttr, nn
from paddle.nn import functional as F
import numpy as np


def get_para_bias_attr(l2_decay, k, name):
    regularizer = paddle.regularizer.L2Decay(l2_decay)
    stdv = 1.0 / math.sqrt(k * 1.0)
    initializer = nn.initializer.Uniform(-stdv, stdv)
    weight_attr = ParamAttr(
        regularizer=regularizer, initializer=initializer, name=name + "_w_attr")
    bias_attr = ParamAttr(
        regularizer=regularizer, initializer=initializer, name=name + "_b_attr")
    return [weight_attr, bias_attr]

class MTLHead(nn.Layer):
    def __init__(self, in_channels, out_channels_ctc, out_channels_att,hidden_size,fc_decay=0.0004, **kwargs):
        super(MTLHead, self).__init__()
        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=fc_decay, k=in_channels, name='ctc_fc')
        self.fc = nn.Linear(
            in_channels,
            out_channels_ctc,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            name='ctc_fc')
        self.out_channels_ctc = out_channels_ctc

        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels_att

        self.attention_cell = AttentionGRUCell(
            in_channels, hidden_size, out_channels_att, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels_att)
    
    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=28):
        #ctc
        predicts = self.fc(inputs)
        # print("!!!!!!!ctc shape:",predicts.shape)
        if not self.training:
            # print("!!!!!!!ctc shape:",predicts.shape)
            predicts = F.softmax(predicts, axis=2)
        #print(predicts)

        #attention
        batch_size = paddle.shape(inputs)[0]
        num_steps = batch_max_length

        hidden = paddle.zeros((batch_size, self.hidden_size))
        output_hiddens = []

        if targets is not None:
            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets[:, i], onehot_dim=self.num_classes)
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs,
                                                               char_onehots)
                output_hiddens.append(paddle.unsqueeze(outputs, axis=1))
            output = paddle.concat(output_hiddens, axis=1)
            probs = self.generator(output)

        else:
            targets = paddle.zeros(shape=[batch_size], dtype="int32")
            probs = None
            char_onehots = None
            outputs = None
            alpha = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes)
                (outputs, hidden), alpha = self.attention_cell(hidden, inputs,
                                                               char_onehots)
                probs_step = self.generator(outputs)
                
                if not self.training:
                    # print("!!!!!!!att2222222222 shape:",probs_step.shape)
                    probs_step = F.softmax(probs_step, axis=1)
                if probs is None:
                    probs = paddle.unsqueeze(probs_step, axis=1)
                else:
                    probs = paddle.concat(
                        [probs, paddle.unsqueeze(
                            probs_step, axis=1)], axis=1)
                next_input = probs_step.argmax(axis=1)
                targets = next_input
        
        return predicts,probs



class AttentionGRUCell(nn.Layer):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionGRUCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias_attr=False)

        self.rnn = nn.GRUCell(
            input_size=input_size + num_embeddings, hidden_size=hidden_size)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):

        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = paddle.unsqueeze(self.h2h(prev_hidden), axis=1)

        res = paddle.add(batch_H_proj, prev_hidden_proj)
        res = paddle.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, axis=1)
        alpha = paddle.transpose(alpha, [0, 2, 1])
        context = paddle.squeeze(paddle.mm(alpha, batch_H), axis=1)
        concat_context = paddle.concat([context, char_onehots], 1)

        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha


class AttentionLSTM(nn.Layer):
    def __init__(self, in_channels, out_channels, hidden_size, **kwargs):
        super(AttentionLSTM, self).__init__()
        self.input_size = in_channels
        self.hidden_size = hidden_size
        self.num_classes = out_channels

        self.attention_cell = AttentionLSTMCell(
            in_channels, hidden_size, out_channels, use_gru=False)
        self.generator = nn.Linear(hidden_size, out_channels)

    def _char_to_onehot(self, input_char, onehot_dim):
        input_ont_hot = F.one_hot(input_char, onehot_dim)
        return input_ont_hot

    def forward(self, inputs, targets=None, batch_max_length=28):
        batch_size = inputs.shape[0]
        num_steps = batch_max_length

        hidden = (paddle.zeros((batch_size, self.hidden_size)), paddle.zeros(
            (batch_size, self.hidden_size)))
        output_hiddens = []

        if targets is not None:
            for i in range(num_steps):
                # one-hot vectors for a i-th char
                char_onehots = self._char_to_onehot(
                    targets[:, i], onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs,
                                                    char_onehots)

                hidden = (hidden[1][0], hidden[1][1])
                output_hiddens.append(paddle.unsqueeze(hidden[0], axis=1))
            output = paddle.concat(output_hiddens, axis=1)
            probs = self.generator(output)

        else:
            targets = paddle.zeros(shape=[batch_size], dtype="int32")
            probs = None

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(
                    targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, inputs,
                                                    char_onehots)
                probs_step = self.generator(hidden[0])
                hidden = (hidden[1][0], hidden[1][1])
                if probs is None:
                    probs = paddle.unsqueeze(probs_step, axis=1)
                else:
                    probs = paddle.concat(
                        [probs, paddle.unsqueeze(
                            probs_step, axis=1)], axis=1)

                next_input = probs_step.argmax(axis=1)

                targets = next_input

        return probs


class AttentionLSTMCell(nn.Layer):
    def __init__(self, input_size, hidden_size, num_embeddings, use_gru=False):
        super(AttentionLSTMCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias_attr=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias_attr=False)
        if not use_gru:
            self.rnn = nn.LSTMCell(
                input_size=input_size + num_embeddings, hidden_size=hidden_size)
        else:
            self.rnn = nn.GRUCell(
                input_size=input_size + num_embeddings, hidden_size=hidden_size)

        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_onehots):
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = paddle.unsqueeze(self.h2h(prev_hidden[0]), axis=1)
        res = paddle.add(batch_H_proj, prev_hidden_proj)
        res = paddle.tanh(res)
        e = self.score(res)

        alpha = F.softmax(e, axis=1)
        alpha = paddle.transpose(alpha, [0, 2, 1])
        context = paddle.squeeze(paddle.mm(alpha, batch_H), axis=1)
        concat_context = paddle.concat([context, char_onehots], 1)
        cur_hidden = self.rnn(concat_context, prev_hidden)

        return cur_hidden, alpha
