import math
import os
from datetime import datetime
from os import listdir
from os.path import isfile, join

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops, dtypes
from tensorflow.python.ops import gen_nn_ops

IMAGE_HEIGHT = 360
IMAGE_WIDTH = 480
IMAGE_DEPTH = 3
BATCH_SIZE = 1


@ops.RegisterGradient("MaxPoolWithArgmax")
def _MaxPoolWithArgmaxGrad(op, grad, unused_argmax_grad):
    return gen_nn_ops._max_pool_grad(op.inputs[0],
                                     op.outputs[0],
                                     grad,
                                     op.get_attr("ksize"),
                                     op.get_attr("strides"),
                                     padding=op.get_attr("padding"),
                                     data_format='NHWC')


def unravel_argmax(argmax, shape):
    output_list = [argmax // (shape[2] * shape[3]),
                   argmax % (shape[2] * shape[3]) // shape[3]]
    return tf.pack(output_list)


def unpool_layer2x2_batch(bottom, argmax):
    bottom_shape = tf.shape(bottom)
    top_shape = [bottom_shape[0], bottom_shape[1] * 2, bottom_shape[2] * 2, bottom_shape[3]]

    batch_size = top_shape[0]
    height = top_shape[1]
    width = top_shape[2]
    channels = top_shape[3]

    argmax_shape = tf.to_int64([batch_size, height, width, channels])
    argmax = unravel_argmax(argmax, argmax_shape)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size * (width // 2) * (height // 2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height // 2, width // 2, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels * (width // 2) * (height // 2)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, height // 2, width // 2, 1])

    t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

    t = tf.concat(4, [t2, t3, t1])
    indices = tf.reshape(t, [(height // 2) * (width // 2) * channels * batch_size, 4])

    x1 = tf.transpose(bottom, perm=[0, 3, 1, 2])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2_argmax(x):
    with tf.device('/gpu:0'):
        return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='SAME')


def get_deconv_filter(f_shape):
    """
      reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    """
    width = f_shape[0]
    heigh = f_shape[0]
    f = math.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)


def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
    return deconv


def conv_pool_layer_with_bias(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    conv = tf.nn.relu(conv2d(input, W) + b)
    return max_pool_2x2_argmax(conv)


def conv_layer_with_bias(input, shape):
    W = weight_variable(shape)
    conv = tf.nn.relu(conv2d(input, W))
    return conv


def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl ** 2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)


def loss(logits, labels):
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, 32))
        epsilon = tf.constant(value=1e-10, name="epsilon")
        logits = tf.add(logits, epsilon)
        label_flat = tf.reshape(labels, (-1, 1))
        with tf.device("/cpu:0"):
            labels = tf.reshape(tf.one_hot(label_flat, depth=32), (-1, 32))
        softmax = tf.nn.softmax(logits)
        cross_entropy = -tf.reduce_sum(labels * tf.log(softmax + epsilon), reduction_indices=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
        tf.add_to_collection('losses', cross_entropy_mean)
        l = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return l


def generate_batch(image, label, min_ex, batch_size, shuffle):
    if shuffle:
        imgs, label_batch = tf.train.shuffle_batch([image, label],
                                                   batch_size=batch_size, num_threads=1,
                                                   capacity=min_ex + 3 * batch_size,
                                                   min_after_dequeue=min_ex)
    else:
        imgs, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=1,
                                           capacity=min_ex + 3 * batch_size)
    return imgs, label_batch


def reader(queue):
    imageName = queue[0]
    labelName = queue[1]
    imageVal = tf.read_file(imageName)
    labelVal = tf.read_file(labelName)
    imageB = tf.image.decode_png(imageVal)
    labelB = tf.image.decode_png(labelVal)
    image = tf.reshape(imageB, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    label = tf.reshape(labelB, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    return image, label


def CamVid(images, labels, batch_size):
    img = ops.convert_to_tensor(images, dtype=dtypes.string)
    lbl = ops.convert_to_tensor(labels, dtype=dtypes.string)

    queue = tf.train.slice_input_producer([img, lbl], shuffle=True)
    image, label = reader(queue)
    reshaped = tf.cast(image, tf.float32)
    min_ex = int(0.2 * 367)
    return generate_batch(reshaped, label, min_ex, batch_size, shuffle=True)


def main(imageN, labelN):
    global_step = tf.Variable(0, trainable=False)
    imgs, labels = CamVid(imageN, labelN, BATCH_SIZE)
    with tf.device('/gpu:0'):
        x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3])
        y_ = tf.placeholder(tf.uint8, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

        pool1, pool1_indices = conv_pool_layer_with_bias(x, [7, 7, IMAGE_DEPTH, 64])
        pool2, pool2_indices = conv_pool_layer_with_bias(pool1, [7, 7, 64, 64])
        pool3, pool3_indices = conv_pool_layer_with_bias(pool2, [7, 7, 64, 64])
        pool4, pool4_indices = conv_pool_layer_with_bias(pool3, [7, 7, 64, 64])

        up4 = deconv_layer(pool4, [2, 2, 64, 64], [BATCH_SIZE, IMAGE_HEIGHT // 8, IMAGE_WIDTH // 8, 64], name="up4")
        de4 = conv_layer_with_bias(up4, [7, 7, 64, 64])
        up3 = deconv_layer(de4, [2, 2, 64, 64], [BATCH_SIZE, IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4, 64], name="up3")
        de3 = conv_layer_with_bias(up3, [7, 7, 64, 64])
        up2 = deconv_layer(de3, [2, 2, 64, 64], [BATCH_SIZE, IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2, 64], name="up2")
        de2 = conv_layer_with_bias(up2, [7, 7, 64, 64])
        up1 = deconv_layer(de2, [2, 2, 64, 64], [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 64], name="up1")
        de1 = conv_layer_with_bias(up1, [7, 7, 64, 64])
        kernel = tf.get_variable('weights', [1, 1, 64, 32], initializer=msra_initializer(1, 64))
        conv = tf.nn.conv2d(de1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = bias_variable([32])
        conv_classifier = tf.nn.bias_add(conv, biases)
        cross_entropy = tf.reduce_mean(loss(conv_classifier, y_))
        opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy, global_step)
    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.histogram_summary(var.op.name, var)
    correct = tf.equal(tf.argmax(conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.summary.image('annotated', y_)
    tf.summary.image('input', x)
    tf.summary.scalar('accuracy', accuracy)
    print("Done setting up graph.")
    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.histogram('biases', biases)
    saver = tf.train.Saver(tf.global_variables())
    merged = tf.summary.merge_all()
    now = datetime.now()
    with tf.device('/gpu:0'):
        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter('train/' + now.strftime("%Y%m%d-%H%M%S") + "/", sess.graph)
            test_writer = tf.summary.FileWriter('test')
            print("Step")
            init = tf.global_variables_initializer()
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for step in range(MAX_STEPS):
                imageB, labelB = sess.run([imgs, labels])
                feed = {x: imageB, y_: labelB}
                _, l, a, summary = sess.run([opt, cross_entropy, accuracy, merged], feed_dict=feed)
                print(step, l, a)
                if step % 10 == 0:
                    train_writer.add_summary(summary, step)
                if step % 500 == 0:
                    saver.save(sess, os.path.join(os.curdir, 'model.ckpt'), global_step=global_step)
            coord.request_stop()
            coord.join(threads)


MAX_STEPS = 10000
NUM_CLASSES = 32
LEARNING_RATE = 1e-3

if __name__ == "__main__":
    train = "CamVid/train/"
    annot = "CamVid/trainannot/"
    images = [train + f for f in listdir(train) if isfile(join(train, f))]
    annotated = [annot + f for f in listdir(annot) if isfile(join(annot, f))]
    main(images, annotated)
