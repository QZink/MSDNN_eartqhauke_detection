import tensorflow.contrib.slim as slim
import tensorflow as tf

trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

class config():
    def __init__(self):
        self.D_TYPE = tf.float32
        self.input_height = 1
        self.input_width = 5000
        self.input_channals = 3
        self.batch_size = 8
        self.class_number = 2
        self.learning_rate = 0.02
        self.regularizer_decay = 0.001
        self.initializer_stddev = 0.1
        self.batch_norm_decay = 0



Config = config()

def SLC_arg_scope(weight_decay=Config.regularizer_decay, stddev=Config.initializer_stddev):
    batch_norm_params = {
        "decay": Config.batch_norm_decay,
        "epsilon": 0.0001, "updates_collections": tf.GraphKeys.UPDATE_OPS,
        "zero_debias_moving_mean": True
    }
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=trunc_normal(stddev=stddev),
                        activation_fn=tf.nn.relu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params
                        ) as scope:
        return scope

def channal_compare_cnn(input_1,input_2,channals=32,name='channal_compare'):
    with tf.variable_scope(name):
        net = tf.concat([input_1, input_2], 3)
        net = slim.conv2d(net, num_outputs=channals, kernel_size=[1, 1], scope='Conv2d_1x1')
    return net

def cell(compare_net,net,channals=32,scope='cell'):
    with tf .variable_scope(scope):
        compare_net = channal_compare_cnn(compare_net, net, channals=channals, name='compare_a1')
        net = slim.conv2d(net, num_outputs=channals, kernel_size=[1, 3], scope='Conv2d_b1_1x3')
        compare_net = slim.max_pool2d(compare_net, kernel_size=[1, 2], stride=[1, 2], scope='MaxPool_a2_1x3')
        net = slim.max_pool2d(net, kernel_size=[1, 2], stride=[1, 2], scope='MaxPool_b2_1x3')
    return  compare_net,net

def MSDNN_multitask(inputs, auxiliary_tensor, is_training=True, name='MSDNN_multitask'):
    print('MSDNN multitask set up')
    endpoint = {}
    with tf.variable_scope(name):
        with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=[1, 1], padding="SAME"):
                compare_net = slim.conv2d(inputs, num_outputs=32,kernel_size=[1, 3],scope='Conv2d_a0_1x3')
                net = slim.conv2d(compare_net, num_outputs=32,kernel_size=[1, 3],scope='Conv2d_b0_1x3')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a0')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a1')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a2')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a3')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a4')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a5')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a6')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a7')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a8')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a9')
                net = slim.flatten(compare_net, scope='flat')
                net1 = slim.fully_connected(net, num_outputs=128, weights_initializer=trunc_normal(stddev=0.01),
                                            scope='FC_b0')
                net1=slim.dropout(net1,keep_prob=0.6)
                net1 = slim.fully_connected(net1, num_outputs=2, weights_initializer=trunc_normal(stddev=0.01),
                                            activation_fn=None, normalizer_fn=None, normalizer_params=None,
                                            scope='FC_b1')
                y = tf.nn.softmax(net1,name='output_softmax_main')
                net2 = tf.matmul(auxiliary_tensor, net)
                net2 = slim.fully_connected(net2, num_outputs=128, weights_initializer=trunc_normal(stddev=0.01),
                                           scope='FC_c0')
                net2=slim.dropout(net2,keep_prob=0.6)
                net2 = slim.fully_connected(net2, num_outputs=2, weights_initializer=trunc_normal(stddev=0.01),
                                            activation_fn=None, normalizer_fn=None, normalizer_params=None,
                                            scope='FC_c1')
                sub_y = tf.nn.softmax(net2,name='output_softmax_sub')
    return y, sub_y, endpoint

def MSDNN(inputs, is_training=True, name='MSDNN'):
    print('MSDNN set up')
    endpoint = {}
    with tf.variable_scope(name):
        with slim.arg_scope([slim.dropout, slim.batch_norm], is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=[1, 1], padding="SAME"):
                compare_net = slim.conv2d(inputs, num_outputs=32,kernel_size=[1, 3],scope='Conv2d_a0_1x3')
                net = slim.conv2d(compare_net, num_outputs=32,kernel_size=[1, 3],scope='Conv2d_b0_1x3')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a0')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a1')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a2')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a3')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a4')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a5')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a6')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a7')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a8')
                compare_net,net=cell(compare_net,net,channals=32,scope='v3_cell_a9')
                net = slim.flatten(compare_net, scope='flat')
                net1 = slim.fully_connected(net, num_outputs=128, weights_initializer=trunc_normal(stddev=0.01),
                                            scope='FC_b0')
                net1=slim.dropout(net1,keep_prob=0.6)
                net1 = slim.fully_connected(net1, num_outputs=2, weights_initializer=trunc_normal(stddev=0.01),
                                            activation_fn=None, normalizer_fn=None, normalizer_params=None,
                                            scope='FC_b1')
                y = tf.nn.softmax(net1,name='output_softmax_main')
    return y, endpoint
