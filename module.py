import tensorflow as tf
from ops import *

def generator_gatedcnn(inputs, reuse = False, scope_name = 'generator_gatedcnn'):

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features] for 1D convolution
    inputs = tf.transpose(inputs, perm = [0, 2, 1], name = 'input_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv')
        h1_gates = conv1d_layer(inputs = inputs, filters = 128, kernel_size = 15, strides = 1, activation = None, name = 'h1_conv_gates')
        h1_glu = gated_linear_layer(inputs = h1, gates = h1_gates, name = 'h1_glu')

        # Downsample
        d1 = downsample1d_block(inputs = h1_glu, filters = 256, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block1_')
        d2 = downsample1d_block(inputs = d1, filters = 512, kernel_size = 5, strides = 2, name_prefix = 'downsample1d_block2_')

        # Residual blocks
        r1 = residual1d_block(inputs = d2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block1_')
        r2 = residual1d_block(inputs = r1, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block2_')
        r3 = residual1d_block(inputs = r2, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block3_')
        r4 = residual1d_block(inputs = r3, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block4_')
        r5 = residual1d_block(inputs = r4, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block5_')
        r6 = residual1d_block(inputs = r5, filters = 1024, kernel_size = 3, strides = 1, name_prefix = 'residual1d_block6_')

        # Upsample
        u1 = upsample1d_block(inputs = r6, filters = 1024, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block1_')
        u2 = upsample1d_block(inputs = u1, filters = 512, kernel_size = 5, strides = 1, shuffle_size = 2, name_prefix = 'upsample1d_block2_')

        # Output
        o1 = conv1d_layer(inputs = u2, filters = 24, kernel_size = 15, strides = 1, activation = None, name = 'o1_conv')
        o2 = tf.transpose(o1, perm = [0, 2, 1], name = 'output_transpose')

    return o2


########################################################################################################################
########################################################################################################################


def Style_Encoder(inputs, style_dim=16, reuse=False, scope='style_encoder'):                                                            # [1, 24, 128] = [batch_size, feature_channel, time]

    inputs = tf.transpose(inputs, perm=[0, 2, 1], name='input_transpose')                                                               # [1, 128, 24] = [batch_size, time, feature_channel]

    with tf.variable_scope(scope, reuse=reuse):

        h1 = conv1d_layer(inputs=inputs, filters=128, kernel_size=15, strides=1, name='h1_conv')                                        # [1, 128, 128]
        h1_gates = conv1d_layer(inputs=inputs, filters=128, kernel_size=15, strides=1, name='h1_conv_gates')
        h1_glu = gated_linear_layer(inputs=h1, gates=h1_gates, name='h1_glu')

        # Downsample
        d1 = downsample1d_block_withoutIN(inputs=h1_glu, filters=256, kernel_size=5, strides=2, name_prefix='downsample1d_block1')      # [1, 64, 256]
        d2 = downsample1d_block_withoutIN(inputs=d1, filters=512, kernel_size=5, strides=2, name_prefix='downsample1d_block2')          # [1, 32, 512]

        d3 = downsample1d_block_withoutIN(inputs=d2, filters=512, kernel_size=3, strides=2, name_prefix='downsample1d_block3')          # [1, 16, 512]
        d4 = downsample1d_block_withoutIN(inputs=d3, filters=512, kernel_size=3, strides=2, name_prefix='downsample1d_block4')          # [1, 8, 512]

        # Global Average Pooling
        p1 = adaptive_avg_pooling(d4)                                                                                                   # [1, 1, 512]
        style = conv1d_layer(inputs=p1, filters=style_dim, kernel_size=1, strides=1, name='SE_logit')                                   # [1, 1, 16]

        return style                                                                                                                    # [1, 1, 16]


def Content_Encoder(inputs, reuse=False, scope='content_encoder'):
    # IN removes the original feature mean and variance that represent important style information
    inputs = tf.transpose(inputs, perm=[0, 2, 1], name='input_transpose')                                                               # [1, 24, 128] = [batch_size, time, feature_channel]

    with tf.variable_scope(scope, reuse=reuse):

        h1 = conv1d_layer(inputs=inputs, filters=128, kernel_size=15, strides=1, name='h1_conv')                                        # [1, 128, 128]
        h1_norm = instance_norm_layer(inputs=h1, name='h1_norm')
        h1_gates = conv1d_layer(inputs=inputs, filters=128, kernel_size=15, strides=1, name='h1_gates')
        h1_norm_gates = instance_norm_layer(inputs=h1_gates, name='h1_norm_gates')
        h1_glu = gated_linear_layer(inputs=h1_norm, gates=h1_norm_gates, name='h1_glu')

        # downsample
        d1 = downsample1d_block(inputs=h1_glu, filters=256, kernel_size=5, strides=2, name_prefix='downsample1d_block1')                # [1, 64, 256]
        d2 = downsample1d_block(inputs=d1, filters=512, kernel_size=5, strides=2, name_prefix='downsample1d_block2')                    # [1, 32, 512]

        # Residual blocks
        r1 = residual1d_block(inputs=d2, filters = 512, kernel_size=3, strides=1, name_prefix='residual1d_block1')                     # [1, 32, 512]
        r2 = residual1d_block(inputs=r1, filters = 512, kernel_size=3, strides=1, name_prefix='residual1d_block2')
        r3 = residual1d_block(inputs=r2, filters = 512, kernel_size=3, strides=1, name_prefix='residual1d_block3')
        content = residual1d_block(inputs=r3, filters=512, kernel_size=3, strides=1, name_prefix='residual1d_block4')

        return content                                                                                                                  # [1, 32, 512]


def MLP(style, reuse=False, scope='MLP'):                                                                                               # [1, 1, 16]

    with tf.variable_scope(scope, reuse=reuse):

        x1 = linear(style, 512, scope='linear_1')                                                                                       # [1, 1, 512]
        x1_gates = linear(x1, 512, scope='linear_1_gates')
        x1_glu = gated_linear_layer(inputs=x1, gates=x1_gates, name='x1_glu')

        x2 = linear(x1_glu, 512, scope='linear_2')
        x2_gates = linear(x2, 512, scope='linear_2_gates')
        x2_glu = gated_linear_layer(inputs=x2, gates=x2_gates, name='x2_glu')

        mu = linear(x2_glu, 512, scope='mu')
        sigma = linear(x2_glu, 512, scope='sigma')

        mu = tf.reshape(mu, shape=[-1, 1, 512])                                                                                         # [1, 1, 512]
        sigma = tf.reshape(sigma, shape=[-1, 1, 512])                                                                                   # [1, 1, 512]

        return mu, sigma                                                                                                                # [1, 1, 512]


def Decoder(content, style, reuse=False, scope="decoder"):

    with tf.variable_scope(scope, reuse=reuse):

        mu, sigma = MLP(style, reuse)                                                                                                   # [1, 1, 512]
        x = content                                                                                                                     # [1, 32, 512]

        # Adaptive Residual blocks
        r1 = residual1d_block_adaptive(inputs=x, filters=512, mu=mu, sigma=sigma, kernel_size=3, strides=1, name_prefix='residual1d_block1')        # [1, 32, 512]
        r2 = residual1d_block_adaptive(inputs=r1, filters=512, mu=mu, sigma=sigma, kernel_size=3, strides=1, name_prefix='residual1d_block2')
        r3 = residual1d_block_adaptive(inputs=r2, filters=512, mu=mu, sigma=sigma, kernel_size=3, strides=1, name_prefix='residual1d_block3')

        # Upsample
        u1 = upsample1d_block(inputs=r3, filters=512, kernel_size=5, strides=1, shuffle_size=2, name_prefix='upsample1d_block1')        # [1, 64, 512]
        u2 = upsample1d_block(inputs=u1, filters=256, kernel_size=5, strides=1, shuffle_size=2, name_prefix='upsample1d_block2')        # [1, 128, 256]

        # Output
        o1 = conv1d_layer(inputs=u2, filters=24, kernel_size=15, strides=1, name='o1_conv')                                             # [1, 128, 24]
        o2 = tf.transpose(o1, perm=[0, 2, 1], name='output_transpose')                                                                  # [1, 24, 128]

        return o2                                                                                                                       # [1, 24, 128] = [batch_size, feature_channel, time]


def discriminator(inputs, reuse=False, scope='discriminator'):

    # inputs = [batch_size, num_features, time]
    # add channel for 2D convolution [batch_size, num_features, time, 1]
    inputs = tf.expand_dims(inputs, -1)                                                                                                 # [1, 24, 128, 1]

    with tf.variable_scope(scope, reuse=reuse):

        h1 = conv2d_layer(inputs=inputs, filters=128, kernel_size=[3, 3], strides=[1, 2], name='h1_conv')                               # [1, 24, 64, 128]
        h1_gates = conv2d_layer(inputs=inputs, filters=128, kernel_size=[3, 3], strides=[1, 2], name='h1_conv_gates')
        h1_glu = gated_linear_layer(inputs=h1, gates=h1_gates, name='h1_glu')

        # Downsample
        d1 = downsample2d_block(inputs=h1_glu, filters=256, kernel_size=[3, 3], strides=[2, 2], name_prefix='downsample2d_block1')      # [1, 12, 32, 256]
        d2 = downsample2d_block(inputs=d1, filters=512, kernel_size=[3, 3], strides=[2, 2], name_prefix='downsample2d_block2')          # [1, 6, 16, 512]
        d3 = downsample2d_block(inputs=d2, filters=1024, kernel_size=[6, 3], strides=[1, 2], name_prefix='downsample2d_block3')         # [1, 6, 8, 1024]

        # Output
        o1 = tf.layers.dense(inputs=d3, units=1, activation=tf.nn.sigmoid)

        return o1                                                                                                                       # [1, 6, 8, 1]
