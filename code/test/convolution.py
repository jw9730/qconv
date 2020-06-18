import os
import sys
import pickle
import numpy as np
import struct
from dnn_vec import DnnGraphBuilder, DnnInferenceEngine

class CONV(object):

    def __init__(self, in_shape, kernel):
        self.kernel = kernel
        self.g = DnnGraphBuilder()
        self.build_graph(in_shape)
        self.sess = DnnInferenceEngine(self.g)


    def build_graph(self, in_shape):
        k_w = self.kernel

        inp = self.g.create_input(in_shape)

        out = self.g.create_conv2d(inp, k_w, strides=[1, 1, 1, 1], padding='SAME')

        self.g.set_out_node(out) 

    def inference(self, im):
        out = self.sess.run(im)
        return out 
