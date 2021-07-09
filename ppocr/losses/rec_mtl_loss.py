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
from numpy import angle

import paddle
from paddle import nn

class MTLLoss(nn.Layer):
    def __init__(self, **kwargs):
        super(MTLLoss, self).__init__()
        self.loss_func_ctc = nn.CTCLoss(blank=0, reduction='none')
        self.loss_func_att = nn.CrossEntropyLoss(weight=None, reduction='none')
       
    def __call__(self, predicts,probs, batch):
        predicts = predicts.transpose((1, 0, 2))
        N, B, _ = predicts.shape
        preds_lengths = paddle.to_tensor([N] * B, dtype='int64')
        labels = batch[1].astype("int32")
        # print("labels::::",labels.shape)
        label_lengths = batch[2].astype('int64')
        loss = self.loss_func_ctc(predicts, labels, preds_lengths, label_lengths)
        # print(loss.shape)
        loss = loss.mean()  # sum

        targets = batch[1].astype("int64")
        batch_num = targets.shape[0]
        label_lengths = batch[2].astype('int64')
        # print("111",targets.shape)
        # mask = paddle.zeros(targets.shape)
        mask =  targets > 1 
        # print("mask: ",mask.astype(int))
        mask = paddle.reshape(mask.astype(int),[-1])
        # print("mask shape :",mask.shape)
        # print("& the show: ",mask.numpy() & targets.numpy())
        
        batch_size, num_steps, num_classes = probs.shape[0], probs.shape[
            1], probs.shape[2]
        assert len(targets.shape) == len(list(probs.shape)) - 1, \
            "The target's shape and inputs's shape is [N, d] and [N, num_steps]"

        inputs = paddle.reshape(probs, [-1, probs.shape[-1]])
        targets = paddle.reshape(targets, [-1])
        # print(self.loss_func_att(inputs, targets).shape)
        # print(self.loss_func_att(inputs, targets)*mask)
        return {'loss_ctc': loss,'loss_att': paddle.sum((self.loss_func_att(inputs, targets)))/batch_num}
        # return {'loss_ctc': loss,'loss_att': (self.loss_func_att(inputs, targets))}
