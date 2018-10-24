"""The Model Implementation of Neural Arithmetic Logical Unit"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn


class NAC(nn.Block):
    def __init__(self, in_units, units):
        super(NAC, self).__init__()

        self.W_hat = self.params.get('W_hat', shape=(in_units, units))
        self.M_hat = self.params.get('M_hat', shape=(in_units, units))

    def forward(self, x):
        W = nd.tanh(self.W_hat.data()) * nd.sigmoid(self.M_hat.data())
        return nd.dot(x, W)


class NALU(nn.Block):
    def __init__(self, in_units, units):
        super(NALU, self).__init__()

        self.W0_hat = self.params.get('W0_hat', shape=(in_units, units))
        self.M0_hat = self.params.get('M0_hat', shape=(in_units, units))
        self.W1_hat = self.params.get('W1_hat', shape=(in_units, units))
        self.M1_hat = self.params.get('M1_hat', shape=(in_units, units))

        self.dependent_G = False  # whether the gate is dependent on the input

        if self.dependent_G:
            self.G = self.params.get('G', shape=(in_units, units))
        else:
            self.G = self.params.get('G', shape=(units,))

    def forward(self, x):
        if self.dependent_G:
            g = nd.sigmoid(nd.dot(x, self.G.data()))
        else:
            g = nd.sigmoid(self.G.data())

        W0 = nd.tanh(self.W0_hat.data()) * nd.sigmoid(self.M0_hat.data())
        W1 = nd.tanh(self.W1_hat.data()) * nd.sigmoid(self.M1_hat.data())
        a = nd.dot(x, W0)
        m = nd.exp(nd.dot(nd.log(nd.abs(x) + 1e-10), W1))
        y = g * a + (1 - g) * m

        return y

