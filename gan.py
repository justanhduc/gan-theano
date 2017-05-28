import theano
from theano import tensor as T
import numpy as np
import gzip
import cPickle as pickle
from theano.compile.nanguardmode import NanGuardMode
from matplotlib import pyplot as plt
import time
from random import shuffle

import layers
import optimization
import utils
import metrics


def build_gan(training_data, batch_size, k):
    x = T.fmatrix('D_input')
    z = T.fmatrix('G_input')
    step = T.scalar('step', dtype='int32')

    placeholder_x = theano.shared(np.zeros((batch_size, 784), dtype=theano.config.floatX))
    placeholder_z = theano.shared(np.zeros((batch_size, 100), dtype=theano.config.floatX))

    discriminator = [
        layers.FullyConnectedLayer(784, 128, layer_name='D1'),
        layers.FullyConnectedLayer(128, 1, activation=T.nnet.sigmoid, layer_name='D2')
    ]


    generator = [
        layers.FullyConnectedLayer(100, 512, layer_name='G1'),
        layers.FullyConnectedLayer(512, 784, activation=T.nnet.sigmoid, layer_name='G2')
    ]

    G_output = inference(z, generator)
    D_output_real = inference(x, discriminator)
    D_output_fake = inference(G_output, discriminator)

    D_loss = -T.mean(T.log(D_output_real) + T.log(1. - D_output_fake))
    G_loss = -T.mean(T.log(D_output_fake)) #+ metrics.GaussianCrossEntropy(G_output, x)

    D_params = []
    G_params = []
    for i in xrange(len(discriminator)):
        D_params += discriminator[i].params
    for i in xrange(len(generator)):
        G_params = generator[i].params
    D_grads = T.grad(D_loss, D_params)
    G_grads = T.grad(G_loss, G_params)

    D_opt = optimization.Adam(D_params)
    D_deltas = D_opt.deltaXt(D_grads, step)
    D_updates = [(D_param, D_param - D_delta) for D_param, D_delta in zip(D_params, D_deltas)]
    G_opt = optimization.Adam(G_params)
    G_deltas = G_opt.deltaXt(G_grads, step)
    G_updates = [(G_param, G_param - G_delta) for G_param, G_delta in zip(G_params, G_deltas)]

    D_train = theano.function([step], [D_loss, D_output_fake, D_output_real], givens={x: placeholder_x, z: placeholder_z}, on_unused_input='warn',
                              allow_input_downcast=True, updates=D_updates)
    G_train = theano.function([step], G_loss, givens={z: placeholder_z}, on_unused_input='warn',
                              allow_input_downcast=True, updates=G_updates)
    G_sample = theano.function([], G_output, givens={z: placeholder_z}, allow_input_downcast=True)

    num_training_batches = len(training_data) / batch_size
    print 'Total number of batches: %d' % num_training_batches
    # x, y = zip(*training_data)
    plt.figure()
    for epoch in xrange(10000):
        shuffle(training_data)
        print 'Epoch %d starts...' % (epoch + 1)
        batch = utils.generator(zip(*training_data), batch_size)

        D_error = 0.
        fake_confidence = 0.
        real_confidence = 0.
        G_error = 0.
        idx = 0
        for b in batch:
            utils.update_input(b[0], placeholder_x, True)
            utils.update_input(sample(100, b[1], 10), placeholder_z, True, **{'shape_x': (batch_size, 100)})
            derr, f, r = D_train(epoch + 1)
            D_error += derr
            fake_confidence += f
            real_confidence += r

            if idx % k == 0:
                utils.update_input(sample(100, b[1], 10), placeholder_z, True, **{'shape_x': (batch_size, 100)})
                G_error += G_train(epoch + 1)
            idx += 1

        if np.isnan(D_error) or np.isnan(G_error):
            print 'A loss diverges. Training will be terinated.'
            break

        print '\tDiscriminator error is: %.4f /Detecting fake data confidence: %.2f /Detecting real data confidence: %.2f'\
              % (D_error / num_training_batches, np.mean(fake_confidence / num_training_batches), np.mean(real_confidence / num_training_batches))
        print '\tGenerator error is: %.4f ' % (G_error / (num_training_batches/k))

        if epoch % 2 == 0:
            plt.clf()
            label = np.random.randint(0, 10, size=(16,))
            utils.update_input(sample(100, label, 10), placeholder_z, True, **{'shape_x': (16, 100)})
            output = G_sample()
            for idx in xrange(16):
                ax = plt.subplot(4, 4, idx+1)
                img = output[idx]
                img = np.reshape(img, (28, 28))
                ax.set_title('%d' % label[idx])
                plt.axis('off')
                plt.imshow(img, cmap='gray')
            plt.show(block=False)
            plt.pause(1e-5)


def inference(input, model):
    feed = input
    for layer in model:
        feed = layer.get_output(feed)
    return feed


def sample(n, range, numel):
    range = np.asarray(range)
    s = []
    for r in range:
        s.append(np.random.uniform(r*numel/100., (r+1)*numel/100., size=(n,)))
    s = np.asarray(s, dtype='float32')
    return s

if __name__ == '__main__':
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()

    training_set = zip(train_set[0], train_set[1]) + zip(valid_set[0], valid_set[1]) + zip(test_set[0], test_set[1])
    build_gan(training_set, 500, 2)
