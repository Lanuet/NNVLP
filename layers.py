from lasagne.layers import Layer
from lasagne import init


class StaticEmbeddingLayer(Layer):
    def __init__(self, incoming, input_size, output_size, W=init.Normal(), **kwargs):
        super(StaticEmbeddingLayer, self).__init__(incoming, **kwargs)

        self.input_size = input_size
        self.output_size = output_size

        self.W = self.add_param(W, (input_size, output_size), name="W", trainable=False)

    def get_output_shape_for(self, input_shape):
        return input_shape + (self.output_size, )

    def get_output_for(self, input, **kwargs):
        return self.W[input]