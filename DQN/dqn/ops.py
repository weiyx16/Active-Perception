import tensorflow as tf

def clipped_error(x):
    """
        # Huber loss (delta = 1)
        L(a) = if abs(a) < delta -> 0.5*a*a
               else -> delta*(abs(a) - 0.5*delta)
    """
    # function where return the coordination of the values which is meet for the condition
    # where(condition, x, y) 根据condition返回x或y中的元素
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

def conv2d(x,
           output_dim,
           kernel_size = [3, 3],
           stride = [1, 1],
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='conv2d'):
    """
        conv2d layer with self-defined params
    """
    with tf.variable_scope(name):
        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            # Notice the third params is the channel number that is -> C
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)
        
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)
        # TODO: 这个return放在with外面还是里面有区别吗？源码编写方式有点奇怪
        if activation_fn != None:
            out = activation_fn(out)

        return out, w, b

def max_pool(x, 
            kernel_size = [2, 2],
            stride = [1, 1],
            data_format='NHWC',
            padding='VALID', 
            name = 'maxpool'):
    """
        max pooling layer with self-defined params
    """
    with tf.variable_scope(name):
        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            # Notice the third params is the channel number that is -> C
            kernel_shape = [1, 1, kernel_size[0], kernel_size[1]]
        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [1, kernel_size[0], kernel_size[1], 1]
        
        out = tf.nn.max_pool(x, ksize=kernel_shape, strides=stride, padding=padding, data_format=data_format)
        return out

def deconv2d(x,
           stride,
           kernel_size = [2, 2],
           initializer=tf.contrib.layers.xavier_initializer(),
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='VALID',
           name='deconv2d'):
    """
        deconv2d layer with self-defined params
    """
    
    input_shape = x.get_shape().as_list()
    with tf.variable_scope(name):
        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            # Notice the third params is the channel number that is -> C
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], input_shape[1]//2]
            output_shape = tf.stack([input_shape[0], input_shape[1]//2, input_shape[2]*2, input_shape[3]*2]) 

        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], input_shape[3]//2]
            output_shape = tf.stack([input_shape[0], input_shape[1]*2, input_shape[2]*2, input_shape[3]//2]) 

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        out = tf.nn.conv2d_transpose(x, w, output_shape, stride, padding, data_format=data_format)
        
        if activation_fn != None:
            out = activation_fn(out)
        return out, w

def crop_and_concat(down_sample, up_sample, data_format = 'NHWC', name='crop_and_concat'):
    """
        Cascade downsample and upsample tensors
    """
    with tf.variable_scope(name):
        down_sample_shape = down_sample.get_shape().as_list()
        up_sample_shape = up_sample.get_shape().as_list()
        # offsets for the top left corner of the crop
        if data_format == 'NCHW':
            offsets = [0, 0, (down_sample_shape[2] - up_sample_shape[2]) // 2, (down_sample_shape[3] - up_sample_shape[3]) // 2]
            size = [-1, -1, up_sample_shape[2], up_sample_shape[3]]
            down_sample_crop = tf.slice(down_sample, offsets, size)
            return tf.concat(1, [down_sample_crop, up_sample])
        elif data_format == 'NHWC':
            offsets = [0, (down_sample_shape[1] - up_sample_shape[1]) // 2, (down_sample_shape[2] - up_sample_shape[2]) // 2, 0]
            size = [-1, up_sample_shape[1], up_sample_shape[2], -1]
            down_sample_crop = tf.slice(down_sample, offsets, size)
            return tf.concat(3, [down_sample_crop, up_sample])

def linear(input_, output_size, stddev=0.02, bias_start=0.0, activation_fn=None, name='linear'):
    """
        Dense layer with self-defined params
        # TODO: We can add initializer function params in the input params
    """
    # shape is batch_size * all_params_list
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                initializer=tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias', [output_size], tf.float32,
                initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn != None:
            out = activation_fn(out)

        return out, w, b
