"""Show how neural networks fail to learn (extrapolate) an identity function"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon, autograd, nd
from mxnet.gluon import nn

class Net(nn.Block):
    def __init__(self, act):
        super(Net, self).__init__()

        with self.name_scope():
            self.dense1 = nn.Dense(8)
            self.dense2 = nn.Dense(8)
            self.dense3 = nn.Dense(1)

        self.act = act

    def forward(self, x):
        x = self.act(self.dense1(x))
        x = self.act(self.dense2(x))
        x = self.dense3(x)
        return x


def train(net, low, high, batch_size, n_batch):
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    l2loss = gluon.loss.L2Loss()

    moving_loss = None
    print_every = 1000

    for i in range(n_batch):
        x = np.random.uniform(low, high, size=(batch_size, 1))

        x = mx.nd.array(x, ctx=ctx)

        with autograd.record():
            y = net(x)
            loss = l2loss(x, y)

        loss.backward()
        trainer.step(batch_size)

        loss = mx.nd.mean(loss).asscalar()
        moving_loss = loss if moving_loss is None else moving_loss * 0.98 + loss * 0.02

        if i % print_every == 0:
            print("Batch: %d\tLoss: %.6f" % (i, moving_loss))


def evaluate(net, x):
    x = mx.nd.array(x, ctx=ctx)
    y = net(x)

    x, y = x.asnumpy().squeeze(), y.asnumpy().squeeze()
    return np.abs(x - y)


if __name__ == '__main__':
    ctx = mx.cpu()

    train_x = (-5, 5)
    test_x = np.arange(-20, 20, 0.1)

    n_model = 20
    n_batch = 5000
    batch_size = 64

    activations = {
        'ReLU': mx.nd.relu,
        'Sigmoid': mx.nd.sigmoid,
        'Tanh': mx.nd.tanh,
        'Relu6': lambda x: mx.nd.clip(mx.nd.relu(x), 0, 6),
        'LeakyRelu': mx.nd.LeakyReLU,
        'ELU': nn.ELU(),
        'SELU': nn.SELU(),
        'PReLU': nn.PReLU(),
        'Swish': nn.Swish(),
    }

    legends = []
    for act in activations:
        test_err = np.zeros_like(test_x)

        for i in range(n_model):
            print("Train:  %s %d/%d" % (act, i+1, n_model))
            net = Net(act=activations[act])
            net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

            train(net, train_x[0], train_x[1], batch_size, n_batch)
            err = evaluate(net, test_x)

            test_err += err
        plt.plot(test_x, test_err / n_model)
        legends.append(act)

    plt.legend(legends)
    plt.grid()
    plt.ylabel('Mean Absolute Error')
    plt.savefig('failure.png')
    plt.show()

