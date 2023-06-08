import os
print(os.getcwd())

import mxnet as mx
from torch import nn
import torch
from mxnet.rnn import BaseRNNCell
import numpy as np

from nowcasting.ops import *
from nowcasting.operators.conv_rnn import BaseConvRNN

class ConvLSTM(BaseConvRNN):
    def __init__(self, num_filter, b_h_w,
                 h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                 i2h_kernel=(3, 3), i2h_stride=(1, 1),
                 i2h_pad=(1, 1), i2h_dilate=(1, 1),
                 act_type="tanh", prefix="ConvLSTM", lr_mult=1.0):
        super(ConvLSTM, self).__init__(num_filter=num_filter,
                                        b_h_w=b_h_w,
                                        h2h_kernel=h2h_kernel,
                                        h2h_dilate=h2h_dilate,
                                        i2h_kernel=i2h_kernel,
                                        i2h_pad=i2h_pad,
                                        i2h_dilate=i2h_dilate,
                                        act_type=act_type,
                                        prefix=prefix)
        self._act_type = "tanh"
        print(self._act_type)
        self.i2h_weight = self.params.get("i2h_weight", lr_mult=lr_mult)
        self.i2h_bias = self.params.get("i2h_bias", lr_mult=lr_mult)
        self.h2h_weight = self.params.get("h2h_weight", lr_mult=lr_mult)
        self.h2h_bias = self.params.get("h2h_bias", lr_mult=lr_mult)

    @property
    def state_postfix(self):
        return ['c', 'h']

    @property
    def state_info(self):
        return [{'shape': (self._batch_size, self._num_filter,
                       self._state_height, self._state_width),
             '__layout__': "NCHW"},
            {'shape': (self._batch_size, self._num_filter,
                       self._state_height, self._state_width),
             '__layout__': "NCHW"}]

    def __call__(self, inputs, states=None, is_initial=False, ret_mid=False):
        name = '%s_t%d' % (self._prefix, self._counter)
        self._counter += 1
        if states is None:
            states = self.begin_state()
        states = list(states)
        print(states)
        assert len(states) == 2
        state_h, state_c = states

        if inputs is not None:
            i2h = mx.sym.Convolution(data=inputs,
                                     weight=self.i2h_weight,
                                     bias=self.i2h_bias,
                                     kernel=self._i2h_kernel,
                                     stride=self._i2h_stride,
                                     dilate=self._i2h_dilate,
                                     pad=self._i2h_pad,
                                     num_filter=self._num_filter * 4,
                                     name="%s_i2h" % name)
        else:
            i2h = None

        h2h = mx.sym.Convolution(data=state_h,
                                 weight=self.h2h_weight,
                                 bias=self.h2h_bias,
                                 kernel=self._h2h_kernel,
                                 stride=(1, 1),
                                 dilate=self._h2h_dilate,
                                 pad=self._h2h_pad,
                                 num_filter=self._num_filter * 4,
                                 name="%s_h2h" % name)

        gates = mx.sym.Activation(i2h + h2h, act_type="sigmoid",
                                  name="%s_gates" % name)

        slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                          name="%s_slice" % name)

        in_gate = slice_gates[0]
        forget_gate = slice_gates[1]
        in_transform = slice_gates[2]
        out_gate = slice_gates[3]

        next_c = (forget_gate * state_c) + (in_gate * in_transform)
        next_h = out_gate * mx.sym.Activation(next_c, act_type=self._act_type)

        if ret_mid:
            return next_h, [next_h, next_c], i2h
        else:
            return next_h, [next_h, next_c]



if __name__ == '__main__':
    data = mx.sym.Variable('data')
    data = mx.sym.SliceChannel(data, axis=0, num_outputs=11, squeeze_axis=True)
    conv_lstm1 = ConvLSTM(num_filter=100, b_h_w=(3, 40, 40), prefix="conv_lstm1")
    out, states = conv_lstm1(inputs=data[0], is_initial=True)
    for i in range(1, 11):
        out, states = conv_lstm1(inputs=data[i], states=states)
    conv_gru_forward_backward_time =\
        mx.test_utils.check_speed(out,
                                  location={'data': np.random.normal(size=(11, 3, 128, 40, 40))},
                                  N=2)
    net = mx.mod.Module(out, data_names=['data',], label_names=None, context=mx.gpu())
    net.bind(data_shapes=[('data', (11, 3, 128, 40, 40))],
             grad_req='add')
    net.init_params()
    net.forward(mx.io.DataBatch(data=[mx.random.normal(shape=(11, 3, 128, 40, 40))], label=None), is_train=False)
    print(net.get_outputs()[0].asnumpy())

    # Test ConvLSTM
    data = mx.sym.Variable('data')
    data = mx.sym.SliceChannel(data, axis=0, num_outputs=11, squeeze_axis=True)
    conv_rnn1 = ConvLSTM(num_filter=100, b_h_w=(3, 40, 40),
                        prefix="conv_lstm1")
    out, states = conv_rnn1(inputs=data[0], is_initial=True)
    for i in range(1, 11):
        out, states = conv_rnn1(inputs=data[i], states=states)
    conv_rnn_forward_backward_time = \
        mx.test_utils.check_speed(out,
                                  location={'data': np.random.normal(size=(11, 3, 128, 40, 40))},
                                  N=2)
    net = mx.mod.Module(out, data_names=['data', ], label_names=None, context=mx.gpu())
    net.bind(data_shapes=[('data', (11, 3, 128, 40, 40))],
             grad_req='add')
    net.init_params()
    net.forward(mx.io.DataBatch(data=[mx.random.normal(shape=(11, 3, 128, 40, 40))], label=None),
                is_train=False)
    print(net.get_outputs()[0].asnumpy())

    print("ConvGRU Time:", conv_gru_forward_backward_time)
    print("ConvRNN Time:", conv_rnn_forward_backward_time)
