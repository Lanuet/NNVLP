import lasagne
import theano.tensor as T
import theano
from lasagne.layers import Gate, InputLayer, EmbeddingLayer, DimshuffleLayer, reshape
import lasagne.nonlinearities as nonlinearities
import utils
import time
import sys
import numpy as np
import subprocess
import shlex
from lasagne import init
from lasagne.layers import MergeLayer
from layers import StaticEmbeddingLayer
import config
from utils2 import ObjectDict, make_dict


class CRFLayer(MergeLayer):
    def __init__(self, incoming, num_labels, mask_input=None, W=init.GlorotUniform(), b=init.Constant(0.), **kwargs):
        self.input_shape = incoming.output_shape
        incomings = [incoming]
        self.mask_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = 1
        super(CRFLayer, self).__init__(incomings, **kwargs)
        self.num_labels = num_labels + 1
        self.pad_label_index = num_labels
        num_inputs = self.input_shape[2]
        self.W = self.add_param(W, (num_inputs, self.num_labels, self.num_labels), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (self.num_labels, self.num_labels), name="b", regularizable=False)

    def get_output_shape_for(self, input_shapes):
        input_shape = input_shapes[0]
        return input_shape[0], input_shape[1], self.num_labels, self.num_labels

    def get_output_for(self, inputs, **kwargs):
        input = inputs[0]
        mask = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        out = T.tensordot(input, self.W, axes=[[2], [0]])
        if self.b is not None:
            b_shuffled = self.b.dimshuffle('x', 'x', 0, 1)
            out = out + b_shuffled
        if mask is not None:
            mask_shuffled = mask.dimshuffle(0, 1, 'x', 'x')
            out = out * mask_shuffled
        return out


def build_model(data):
    # create target layer
    target_var = T.imatrix(name='targets')

    # create mask layer
    mask_var = T.matrix(name='masks', dtype=theano.config.floatX)
    layer_mask = InputLayer(shape=(None, data.max_sen_len), input_var=mask_var, name='mask')

    # create word input layer
    word_input_var = T.imatrix(name='word_inputs')
    layer_word_input = InputLayer(shape=(None, data.max_sen_len), input_var=word_input_var, name='word_input')
    layer_word_embedding = StaticEmbeddingLayer(layer_word_input, input_size=data.word_embeddings.shape[0], output_size=data.word_embeddings.shape[1], W=data.word_embeddings, name='word_embedding')

    # create pos input layer
    pos_input_var = T.imatrix(name='pos_inputs')
    layer_pos_input = InputLayer(shape=(None, data.max_sen_len), input_var=pos_input_var, name='pos_input')
    layer_pos_embedding = StaticEmbeddingLayer(layer_pos_input, input_size=data.pos_embeddings.shape[0], output_size=data.pos_embeddings.shape[1], W=data.pos_embeddings, name='pos_embedding')

    # create char input layer
    char_input_var = T.itensor3(name='char_inputs')
    layer_char_input = InputLayer(shape=(None, data.max_sen_len, data.max_word_len), input_var=char_input_var, name='char_input')
    layer_char_input = reshape(layer_char_input, (-1, [2]))
    layer_char_embedding = EmbeddingLayer(layer_char_input, input_size=data.char_embeddings.shape[0], output_size=data.char_embeddings.shape[1], name='char_embedding', W=data.char_embeddings)
    layer_char_input = DimshuffleLayer(layer_char_embedding, pattern=(0, 2, 1))

    # create window size
    conv_window = 3
    _, sent_length, _ = layer_word_embedding.output_shape
    if config.dropout:
        layer_char_input = lasagne.layers.DropoutLayer(layer_char_input, p=0.5)
    # construct convolution layer
    cnn_layer = lasagne.layers.Conv1DLayer(layer_char_input, num_filters=config.num_filters,
                                           filter_size=conv_window, pad='full',
                                           nonlinearity=lasagne.nonlinearities.tanh, name='cnn')
    # infer the pool size for pooling (pool size should go through all time step of cnn)
    _, _, pool_size = cnn_layer.output_shape
    # construct max pool layer
    pool_layer = lasagne.layers.MaxPool1DLayer(cnn_layer, pool_size=pool_size)
    # reshape the layer to match lstm incoming layer [batch * sent_length, num_filters, 1] --> [batch, sent_length,
    # num_filters]
    output_cnn_layer = lasagne.layers.reshape(pool_layer, (-1, sent_length, [1]))
    # finally, concatenate the two incoming layers together.
    incoming = lasagne.layers.concat([output_cnn_layer, layer_word_embedding, layer_pos_embedding], axis=2)
    # create bi-lstm
    if config.dropout:
        incoming = lasagne.layers.DropoutLayer(incoming, p=0.5)
    ingate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                          W_cell=lasagne.init.Uniform(range=0.1))
    outgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                              W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_forward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                        nonlinearity=nonlinearities.tanh)
    lstm_forward = lasagne.layers.LSTMLayer(incoming, config.num_units, mask_input=layer_mask,
                                            grad_clipping=config.grad_clipping, nonlinearity=nonlinearities.tanh,
                                            peepholes=config.peepholes, ingate=ingate_forward, outgate=outgate_forward,
                                            forgetgate=forgetgate_forward, cell=cell_forward, name='forward')
    ingate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                           W_cell=lasagne.init.Uniform(range=0.1))
    outgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                            W_cell=lasagne.init.Uniform(range=0.1))
    # according to Jozefowicz et al.(2015), init bias of forget gate to 1.
    forgetgate_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(),
                               W_cell=lasagne.init.Uniform(range=0.1), b=lasagne.init.Constant(1.))
    # now use tanh for nonlinear function of cell, need to try pure linear cell
    cell_backward = Gate(W_in=lasagne.init.GlorotUniform(), W_hid=lasagne.init.GlorotUniform(), W_cell=None,
                         nonlinearity=nonlinearities.tanh)
    lstm_backward = lasagne.layers.LSTMLayer(incoming, config.num_units, mask_input=layer_mask,
                                             grad_clipping=config.grad_clipping, nonlinearity=nonlinearities.tanh,
                                             peepholes=config.peepholes, backwards=True, ingate=ingate_backward,
                                             outgate=outgate_backward, forgetgate=forgetgate_backward,
                                             cell=cell_backward, name='backward')
    # concatenate the outputs of forward and backward RNNs to combine them.
    concat = lasagne.layers.concat([lstm_forward, lstm_backward], axis=2, name="bi-lstm")
    # dropout for output
    if config.dropout:
        concat = lasagne.layers.DropoutLayer(concat, p=0.5)
    # the shape of Bi-LSTM output (concat) is (batch_size, input_length, 2 * num_hidden_units)
    model = CRFLayer(concat, data.num_labels, mask_input=layer_mask)
    energies = lasagne.layers.get_output(model, deterministic=True)
    prediction = utils.crf_prediction(energies)
    prediction_fn = theano.function([word_input_var, pos_input_var, mask_var, char_input_var], [prediction])

    vars = ObjectDict(make_dict(word_input_var, pos_input_var, target_var, mask_var, char_input_var))
    return model, vars, prediction_fn


def train_model(data_train, data_dev, data_test, model, model_name, vars, label_decoder, output_dir):
    print(1)
    num_tokens = vars.mask_var.sum(dtype=theano.config.floatX)
    print(2)
    energies_train = lasagne.layers.get_output(model)
    print(3)
    energies_eval = lasagne.layers.get_output(model, deterministic=True)
    print(4)
    loss_train = utils.crf_loss(energies_train, vars.target_var, vars.mask_var).mean()
    print(5)
    loss_eval = utils.crf_loss(energies_eval, vars.target_var, vars.mask_var).mean()
    print(6)
    _, corr_train = utils.crf_accuracy(energies_train, vars.target_var)
    print(7)
    corr_train = (corr_train * vars.mask_var).sum(dtype=theano.config.floatX)
    print(8)
    prediction_eval, corr_eval = utils.crf_accuracy(energies_eval, vars.target_var)
    print(9)
    corr_eval = (corr_eval * vars.mask_var).sum(dtype=theano.config.floatX)
    print(10)
    params = lasagne.layers.get_all_params(model, trainable=True)
    print(11)
    updates = lasagne.updates.momentum(loss_train, params=params, learning_rate=config.learning_rate, momentum=0.9)
    print(12)
    train_fn = theano.function([vars.word_input_var, vars.pos_input_var, vars.target_var, vars.mask_var, vars.char_input_var],
                               [loss_train, corr_train, num_tokens], updates=updates)
    print(13)
    eval_fn = theano.function([vars.word_input_var, vars.pos_input_var, vars.target_var, vars.mask_var, vars.char_input_var],
                              [loss_eval, corr_eval, num_tokens, prediction_eval])
    print(14)
    num_batches = data_train.num_data / config.batch_size
    print(15)
    best_loss = 1e+12
    best_acc = 0.0
    best_epoch_loss = 0
    best_epoch_acc = 0
    best_loss_test_err = 0.
    best_loss_test_corr = 0.
    best_acc_test_err = 0.
    best_acc_test_corr = 0.
    stop_count = 0
    lr = config.learning_rate
    for epoch in range(1, config.num_epochs + 1):
        print('Epoch %d (learning rate=%.4f, decay rate=%.4f): ' % (epoch, lr, config.decay_rate))
        train_err = 0.0
        train_corr = 0.0
        train_total = 0
        train_inst = 0
        start_time = time.time()
        num_back = 0
        train_batches = 0
        for batch in utils.iterate_minibatches(data_train, batch_size=config.batch_size, shuffle=True):
            print("training batch")
            word_inputs, pos_inputs, targets, masks, char_inputs = batch
            err, corr, num = train_fn(word_inputs, pos_inputs, targets, masks, char_inputs)
            print("training batch done")
            train_err += err * word_inputs.shape[0]
            train_corr += corr
            train_total += num
            train_inst += word_inputs.shape[0]
            train_batches += 1
            time_ave = (time.time() - start_time) / train_batches
            time_left = (num_batches - train_batches) * time_ave
            sys.stdout.write("\b" * num_back)
            log_info = 'train: %d/%d loss: %.4f, acc: %.2f%%, time left (estimated): %.2fs' % (
                min(train_batches * config.batch_size, data_train.num_data), data_train.num_data,
                train_err / train_inst, train_corr * 100 / train_total, time_left)
            sys.stdout.write(log_info)
            num_back = len(log_info)
        # update training log after each epoch
        assert train_inst == data_train.num_data
        sys.stdout.write("\b" * num_back)
        print('train: %d/%d loss: %.4f, acc: %.2f%%, time: %.2fs' % (min(train_batches * data_train.batch_size, data_train.num_data), data_train.num_data, train_err / data_train.num_data, train_corr * 100 / train_total, time.time() - start_time))
        # evaluate performance on dev data
        dev_err = 0.0
        dev_corr = 0.0
        dev_total = 0
        dev_inst = 0
        for batch in utils.iterate_minibatches(data_dev, batch_size=config.batch_size):
            word_inputs, pos_inputs, targets, masks, char_inputs = batch
            err, corr, num, predictions = eval_fn(word_inputs, pos_inputs, targets, masks, char_inputs)
            dev_err += err * word_inputs.shape[0]
            dev_corr += corr
            dev_total += num
            dev_inst += word_inputs.shape[0]
            utils.output_predictions(predictions, targets, masks, output_dir + '/dev%d' % epoch, label_decoder, is_flattened=False)

        print('dev loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (dev_err / dev_inst, dev_corr, dev_total, dev_corr * 100 / dev_total))
        if model_name != 'pos':
            input = open(output_dir + '/dev%d' % epoch)
            p1 = subprocess.Popen(shlex.split("perl conlleval.pl"), stdin=input)
            p1.wait()
        if best_loss < dev_err and best_acc > dev_corr / dev_total:
            stop_count += 1
        else:
            update_loss = False
            update_acc = False
            stop_count = 0
            if best_loss > dev_err:
                update_loss = True
                best_loss = dev_err
                best_epoch_loss = epoch
            if best_acc < dev_corr / dev_total:
                update_acc = True
                best_acc = dev_corr / dev_total
                best_epoch_acc = epoch
            # evaluate on test data when better performance detected
            test_err = 0.0
            test_corr = 0.0
            test_total = 0
            test_inst = 0
            for batch in utils.iterate_minibatches(data_test, batch_size=config.batch_size):
                word_inputs, pos_inputs, targets, masks, char_inputs = batch
                err, corr, num, predictions = eval_fn(word_inputs, pos_inputs, targets, masks, char_inputs)
                test_err += err * word_inputs.shape[0]
                test_corr += corr
                test_total += num
                test_inst += word_inputs.shape[0]
                utils.output_predictions(predictions, targets, masks, output_dir + '/test%d' % epoch, label_decoder, is_flattened=False)

            np.savez('pre-trained-model/' + model_name + '/weights', *lasagne.layers.get_all_param_values(model))
            print('test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (test_err / test_inst, test_corr, test_total, test_corr * 100 / test_total))
            if model_name != 'pos':
                input = open(output_dir + '/test%d' % epoch)
                p1 = subprocess.Popen(shlex.split("perl conlleval.pl"), stdin=input)
                p1.wait()
            if update_loss:
                best_loss_test_err = test_err
                best_loss_test_corr = test_corr
            if update_acc:
                best_acc_test_err = test_err
                best_acc_test_corr = test_corr
        # stop if dev acc decrease patience time straightly.
        if stop_count == config.patience:
            break
        # re-compile a function with new learning rate for training
        lr = config.learning_rate / (1.0 + epoch * config.decay_rate)
        lasagne.updates.momentum(loss_train, params=params, learning_rate=lr, momentum=0.9)
        train_fn = theano.function([vars.word_input_var, vars.pos_input_var, vars.target_var, vars.mask_var, vars.char_input_var],
                                   [loss_train, corr_train, num_tokens],
                                   updates=updates)
    # print(best performance on test data.)
    print("final best loss test performance (at epoch %d)" % (best_epoch_loss))
    print('test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (best_loss_test_err / test_inst, best_loss_test_corr, test_total, best_loss_test_corr * 100 / test_total))
    print("final best acc test performance (at epoch %d)" % (best_epoch_acc))
    print('test loss: %.4f, corr: %d, total: %d, acc: %.2f%%' % (best_acc_test_err / test_inst, best_acc_test_corr, test_total, best_acc_test_corr * 100 / test_total))