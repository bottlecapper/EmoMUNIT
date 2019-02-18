import tensorflow as tf
import tensorflow.contrib as tf_contrib

weight_init = tf_contrib.layers.variance_scaling_initializer() # kaming init for encoder / decoder
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if scope.__contains__("discriminator") :
            weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        else :
            weight_init = tf_contrib.layers.variance_scaling_initializer()

        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

        return x

def linear(x, units, use_bias=True, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

##################################################################################
# Residual-block
##################################################################################

def resblock(x_init, channels, use_bias=True, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

        return x + x_init

def adaptive_resblock(x_init, channels, mu, sigma, use_bias=True, name='adaptive_resblock') :
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = adaptive_instance_norm(x, mu, sigma)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = adaptive_instance_norm(x, mu, sigma)

        return x + x_init

##################################################################################
# Sampling
##################################################################################

def down_sample(x) :
    return tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='SAME')

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def adaptive_avg_pooling(x):
    # global average pooling
    gap = tf.reduce_mean(x, axis=1, keep_dims=True)

    return gap

##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):
    # gamma, beta = style_mean, style_std from MLP

    c_mean, c_var = tf.nn.moments(content, axes=[1], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta


def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)

##################################################################################
# Loss function
##################################################################################

"""
Author use LSGAN
For LSGAN, multiply each of G and D by 0.5.
However, MUNIT authors did not do this.
"""

def discriminator_loss(type, real, fake):
    n_scale = len(real)
    loss = []

    real_loss = 0
    fake_loss = 0

    for i in range(n_scale) :
        if type == 'lsgan' :
            real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
            fake_loss = tf.reduce_mean(tf.square(fake[i]))

        if type == 'gan' :
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))

        loss.append(real_loss + fake_loss)

    return sum(loss)


def generator_loss(type, fake):
    n_scale = len(fake)
    loss = []

    fake_loss = 0

    for i in range(n_scale) :
        if type == 'lsgan' :
            fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 1.0))

        if type == 'gan' :
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake[i]), logits=fake[i]))

        loss.append(fake_loss)


    return sum(loss)


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss


########################################################################################################################
########################################################################################################################

def gated_linear_layer(inputs, gates, name=None):
    activation = tf.multiply(x=inputs, y=tf.sigmoid(gates), name=name)

    return activation


def instance_norm_layer(
        inputs,
        epsilon=1e-06,
        activation_fn=None,
        name=None):
    instance_norm_layer = tf.contrib.layers.instance_norm(
        inputs=inputs,
        epsilon=epsilon,
        activation_fn=activation_fn,
        scope=name)

    return instance_norm_layer


def layer_norm_layer(
        inputs,
        name=None):
    layer_norm_layer = tf_contrib.layers.layer_norm(
        inputs=inputs,
        center=True,
        scale=True,
        scope=name)

    return layer_norm_layer


def conv1d_layer(
        inputs,
        filters,
        kernel_size,
        strides=1,
        padding='same',
        activation=None,
        name=None):

    if name.__contains__("discriminator"):
        weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    else:
        weight_init = tf_contrib.layers.variance_scaling_initializer()

    conv_layer = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=weight_init,
        kernel_regularizer=weight_regularizer,
        name=name)

    return conv_layer


def conv2d_layer(
        inputs,
        filters,
        kernel_size,
        strides,
        padding='same',
        activation=None,
        name=None):

    if name.__contains__("discriminator"):
        weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
    else:
        weight_init = tf_contrib.layers.variance_scaling_initializer()

    conv_layer = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_initializer=weight_init,
        kernel_regularizer=weight_regularizer,
        name=name)

    return conv_layer


def residual1d_block(
        inputs,
        filters=512,
        kernel_size=3,
        strides=1,
        name_prefix='residule_block'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, name=name_prefix+'_h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, name=name_prefix+'_h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix+'_h1_glu')
    h2 = conv1d_layer(inputs=h1_glu, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h2_conv')
    h2_norm = instance_norm_layer(inputs=h2, name=name_prefix+'_h2_norm')
    h3 = inputs + h2_norm

    return h3


def residual1d_block_adaptive(
        inputs,
        filters=512,
        mu=0.0,
        sigma=1.0,
        kernel_size=3,
        strides=1,
        name_prefix='residule_block'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h1_conv')
    h1_norm = adaptive_instance_norm(content=h1, gamma=mu, beta=sigma)
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h1_gates')
    h1_norm_gates = adaptive_instance_norm(content=h1_gates, gamma=mu, beta=sigma)
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix+'_h1_glu')
    h2 = conv1d_layer(inputs=h1_glu, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h2_conv')
    h2_norm = adaptive_instance_norm(content=h2, gamma=mu, beta=sigma)
    h3 = inputs + h2_norm

    return h3


def downsample1d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        name_prefix='downsample1d_block'):

    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, name=name_prefix+'_h1_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, name=name_prefix+'_h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix+'_h1_glu')

    return h1_glu


def downsample1d_block_withoutIN(
        inputs,
        filters,
        kernel_size,
        strides,
        name_prefix='downsample1d_block_withoutIN'):

    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h1_conv')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h1_gates')
    h1_glu = gated_linear_layer(inputs=h1, gates=h1_gates, name=name_prefix + 'h1_glu')

    return h1_glu


def downsample2d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        name_prefix='downsample2d_block'):

    h1 = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h1_conv')
    h1_norm = instance_norm_layer(inputs=h1, name=name_prefix+'_h1_norm')
    h1_gates = conv2d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h1_gates')
    h1_norm_gates = instance_norm_layer(inputs=h1_gates, name=name_prefix+'_h1_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix+'_h1_glu')

    return h1_glu


def upsample1d_block(
        inputs,
        filters,
        kernel_size,
        strides,
        shuffle_size=2,
        name_prefix='upsample1d_block'):
    h1 = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h1_conv')
    h1_shuffle = pixel_shuffler(inputs=h1, shuffle_size=shuffle_size, name=name_prefix+'_h1_shuffle')
    h1_norm = layer_norm_layer(inputs=h1_shuffle, name=name_prefix+'_h1_layer_norm')
    h1_gates = conv1d_layer(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, name=name_prefix+'_h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs=h1_gates, shuffle_size=shuffle_size, name=name_prefix+'_h1_shuffle_gates')
    h1_norm_gates = layer_norm_layer(inputs=h1_shuffle_gates, name=name_prefix+'_h1_layer_norm_gates')
    h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name=name_prefix+'_h1_glu')

    return h1_glu


def pixel_shuffler(inputs, shuffle_size=2, name=None):
    n = tf.shape(inputs)[0]
    w = tf.shape(inputs)[1]
    c = inputs.get_shape().as_list()[2]

    oc = c // shuffle_size
    ow = w * shuffle_size

    outputs = tf.reshape(tensor=inputs, shape=[n, ow, oc], name=name)

    return outputs
