import tensorflow as tf


def normal_initialier(shape, dtype=tf.float32, partition_info=None):
    return tf.truncated_normal(shape, mean=0.0, stddev=0.01, dtype=dtype)


def initializer_from_string(initializer):
    init = initializer.lower()
    if init == 'normal':
        return normal_initialier
    elif init == 'xavier':
        return tf.contrib.layers.xavier_initializer()
    else:
        raise ValueError('Initializer {} not supported'.format(initializer))


def activation_fn_from_string(name_str):
    activation = name_str.lower()
    if activation == 'sigmoid':
        return tf.nn.sigmoid
    elif activation == 'tanh':
        return tf.nn.tanh
    elif activation == 'relu':
        return tf.nn.relu
    elif activation == 'lrelu':
        return lrelu
    elif activation == 'none':
        return None
    else:
        raise ValueError('Activation function {} not supported'.format(activation))


def lrelu(input_, alpha=0.01, name='lrelu'):
    """Leaky Relu with alpha as the slope"""
    return tf.maximum(alpha * input_, input_, name=name)


def dense(input_, hidden_dim, name='dense'):
    """
    Create a fully connected layer with a bias term.

    :param input_: The input tensor of shape (batch_size, input_dim). If the rank is > 2, it will be flattened.
    :param hidden_dim: The number of hidden units to use.
    :param name: An optional name for the operation.
    :return: A Tensor of shape (batch_size, hidden_dim)
    """
    input_shape = input_.get_shape()
    dims = [v.value for v in input_shape[1:]]
    input_dim = 1
    for d in dims:
        input_dim *= d

    input_r = tf.reshape(input_, [-1, input_dim])

    with tf.variable_scope(name):
        w = tf.get_variable(name + '_w',
                            initializer=tf.truncated_normal([input_dim, hidden_dim], 0.0, 0.01, dtype=tf.float32))
        b = tf.get_variable(name + '_b',
                            shape=[hidden_dim],
                            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        out = tf.add(tf.matmul(input_r, w), b)
        return out


def conv2d_strided(input_, filter_h, filter_w, channels_in, channels_out, initializer,
                   pool_factor=(2, 2), name='conv'):
    """
    2D convolution layer with a bias term. `pool_factor` determines the stride. 
    """
    if isinstance(pool_factor, int):
        strides = (pool_factor, pool_factor)
    else:
        assert len(pool_factor) == 2
        strides = pool_factor

    with tf.variable_scope(name):
        shape = [filter_h, filter_w, channels_in, channels_out]
        w = tf.get_variable(name + '_w',
                            initializer=initializer_from_string(initializer),
                            shape=shape)
        b = tf.get_variable(name + '_b',
                            shape=[channels_out],
                            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        conv = tf.nn.conv2d(input_,
                            filter=w,
                            strides=[1, strides[0], strides[1], 1],
                            padding='SAME')
        out = tf.add(conv, b)
        return out


def deconv2d_strided(input_, filter_h, filter_w, channels_in, channels_out, output_shape, initializer,
                     pool_factor=(2, 2), name='deconv', reuse=False):
    """
    2D deconvolution, i.e. transpose of 2D convolution. `pool_factor` determines the stride. If weights are reused,
    this does the exact opposite of `conv2d_strided`. I.e. it first subtracts the bias, then does the deconvolution.
    If the weights are not shared, the bias is added to the output of the deconvolution.
    """
    if isinstance(pool_factor, int):
        strides = (pool_factor, pool_factor)
    else:
        assert len(pool_factor) == 2
        strides = pool_factor

    with tf.variable_scope(name, reuse=reuse):
        shape = [filter_h, filter_w, channels_out, channels_in]
        w = tf.get_variable(name + '_w',
                            initializer=initializer_from_string(initializer),
                            shape=shape)
        bias_dim = channels_in if reuse else channels_out
        b = tf.get_variable(name + '_b',
                            shape=[bias_dim],
                            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        minus_b = tf.subtract(input_, b) if reuse else input_

        # dynamic shape of input_
        in_shape = tf.shape(input_)
        out_shape = [in_shape[0], output_shape[0], output_shape[1], channels_out]
        conv = tf.nn.conv2d_transpose(minus_b,
                                     filter=w,
                                     output_shape=tf.pack(out_shape),
                                     strides=[1, strides[0], strides[1], 1],
                                     padding='SAME')

        # set the shape of `conv` explicitly again because otherwise all the dimensions are undefined (TF bug)
        true_shape = [s if isinstance(s, int) else None for s in out_shape]
        conv.set_shape(true_shape)
        out = conv if reuse else tf.add(conv, b)
        return out


def conv1d_strided(input_, filter_size, channels_in, channels_out, pool_factor=2, name='conv1d'):
    """
    Performs 1d convolution over the width of the input image. It does so by laying out the height of the input as
    channels and the width of the input as the height and subsequently performing 2D convolution with filters of shape
    [filter_size, 1].
    """
    with tf.variable_scope(name):
        w = tf.get_variable(name + '_w',
                            initializer=tf.truncated_normal([filter_size, 1, channels_in, channels_out],
                                                            mean=0.0, stddev=0.01, dtype=tf.float32))
        b = tf.get_variable(name + '_b',
                            shape=[channels_out],
                            initializer=tf.constant_initializer(0.0, dtype=tf.float32))
        conv = tf.nn.conv2d(input_,
                            filter=w,
                            strides=[1, 1, pool_factor, 1],  # in NCHW format as well
                            padding='SAME',
                            data_format='NCHW')

        # reshape conv so that broadcasting works correctly
        conv_shape = conv.get_shape()
        conv_r = tf.reshape(conv, [-1, 1, conv_shape[2].value, conv_shape[1].value])

        # now add bias and reshape back again
        out_r = tf.add(conv_r, b)
        out = tf.reshape(out_r, [-1, conv_shape[1].value, conv_shape[2].value, 1])
        tf.assert_equal(tf.shape(out), tf.shape(conv))
        return out


def deconv1d_strided(input_, filter_size, channels_in, channels_out, output_height,
                     pool_factor=2, name='deconv1d', reuse=False):
    """
    The transpose of `conv1d_strided`, similar to deconv2d_strided but convolution only performed over the width
    of the image.
    """
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable(name + '_w',
                            initializer=tf.truncated_normal([filter_size, 1, channels_out, channels_in],
                                                            mean=0.0, stddev=0.01, dtype=tf.float32))
        b = tf.get_variable(name + '_b',
                            shape=[channels_in],
                            initializer=tf.constant_initializer(0.0, dtype=tf.float32))

        # reshape input so that broadcasting works correctly
        in_shape = input_.get_shape()
        input_r = tf.reshape(input_, [-1, in_shape[3].value, in_shape[2].value, in_shape[1].value])

        # subtract bias and reshape back again
        minus_b_r = tf.subtract(input_r, b)
        minus_b = tf.reshape(minus_b_r, [-1, in_shape[1].value, in_shape[2].value, in_shape[3].value])

        # compose the output shape
        in_shape = tf.shape(input_)
        out_shape = [in_shape[0], channels_out, output_height, 1]

        # do the convolution
        out = tf.nn.conv2d_transpose(minus_b,
                                     filter=w,
                                     output_shape=tf.pack(out_shape),
                                     strides=[1, 1, pool_factor, 1],
                                     padding='SAME',
                                     data_format='NCHW')

        # set the shape of `out` explicitly again because otherwise all the dimensions are undefined (TF bug)
        out.set_shape([None] + out_shape[1:])
        return out


def unpool(input_, pool_dim, pool_factor, is_training, out_shape=None):
    """
    Unpools a given dimension in the input by a given factor. The un-pooling is done as explained in Holden et al. 2016,
    'A Deep Learning Framework for Character Motion Synthesis and Editing'

    :param input_: Tensor, the input to be unpooled
    :param pool_dim: scalar, the dimension of the input that is to be unpooled
    :param pool_factor: scalar, the factor by which `pool_dim` will be expanded
    :param is_training: If set, unpooler will be 'random', else it will be 'spread'. For 'random' the value to be
      unpooled is randomly set to either the first or second unit that emerges from the unpooling. For 'spread'
      the value is spread across both units evenly.
    :param out_shape: Optional output shape for `pool_dim`. If set, it is made sure that the dimensionality of the
      output tensor is set to `out_shape`.
    :return: the unpooled input in the same format except that `pool_dim` is expanded accordingly
    """
    # move dimension to be unpooled so that it is the last dimension in the input tensor
    n = len(input_.get_shape())
    swap_dims = list(range(0, n))
    swap_dims[pool_dim], swap_dims[-1] = swap_dims[-1], swap_dims[pool_dim]
    input_t = tf.transpose(input_, swap_dims)

    # concatenate along new axis so that result is of size (..., pooled_size, 2).
    cop = tf.pack([input_t, input_t], axis=-1)

    # construct target shape for unpooled Tensor
    target_shape = input_t.get_shape()
    t = [target_shape[i] for i in range(len(target_shape))]
    t[-1] = t[-1] * pool_factor

    # replace first dimension with dynamic shape, because this is the batchsize and usually not known
    t[0] = tf.shape(input_t)[0]

    # for 'random' unpooler
    mask_random = tf.random_uniform(tf.shape(cop))
    mask_random = tf.floor(mask_random / tf.expand_dims(tf.reduce_max(mask_random, axis=n), axis=-1))

    # for 'spread' unpooler
    mask_spread = tf.fill(dims=tf.shape(cop), value=0.5)

    # choose which mask to use depending on `is_training` placeholder
    mask = tf.cond(is_training,
                   lambda: mask_random,
                   lambda: mask_spread)

    # do the actual un-pooling by multiplying mask with cop
    unpooled = tf.reshape(tf.multiply(cop, mask), tf.pack(t))

    # transpose the pool_dim back to its original place
    output = tf.transpose(unpooled, swap_dims)

    # check that output shape is as required
    if out_shape is not None:
        dim_exp = out_shape
        dim_act = output.get_shape()[pool_dim].value
        diff = dim_act - dim_exp
        if diff > 0:
            # Too many dimension, remove the excessive part
            size = [-1] * 4
            size[pool_dim] = dim_exp
            output = tf.slice(output, [0, 0, 0, 0], size)
        elif diff < 0:
            # ignoring this case for now
            pass
        else:
            # no difference, so nothing to do here
            pass

    return output


def batch_normalization(input_, is_training, scope, reuse):
    """
    Adds a batch-normalization layer to the input.
    
    :param input_: A rank 4-tensor in format NHWC.
    :param is_training: A boolean tf.placeholder specifying whether this is a training run or not.
    :param scope: String specifying the scope for the variables being created (or reused). 
    :param reuse: True if variables should be reused. Note that this only effects the ops at training time, at
      test time, we always reuse the variables.
    :return: The batch-normalized inputs.
    """
    # Note: it is important that `updates_collections` is set to None, because this triggers that the required
    # variables are updated in place automatically. It can infer a speed penalty though. If this is crucial, set
    # `updates_collections` to the default, but then remember to set the `updates_collections` as a dependency
    # for the train_op (cf. Tensorflow documentation for details).
    # Also note: not importing tf.contrib.layers.python.layers.batch_norm but using it directly in the code
    # on purpose - importing it causes segfault when used together with `import quaternion`.
    train_bn = tf.contrib.layers.python.layers.batch_norm(
        inputs=input_,
        decay=0.999,
        activation_fn=None,
        updates_collections=None,
        trainable=True,
        scope=scope,
        reuse=reuse)
    test_bn = tf.contrib.layers.python.layers.batch_norm(
        inputs=input_,
        decay=0.999,
        activation_fn=None,
        updates_collections=None,
        trainable=False,
        scope=scope,
        reuse=True)
    out = tf.cond(is_training,
                  lambda: train_bn,
                  lambda: test_bn)
    return out
