import argparse
import math
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


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.get_variable(name="weigh", initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial_value=initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2_argmax(x, name):
    with tf.device('/gpu:0'):
        with tf.variable_scope(name):
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


def conv_pool_layer_with_bias(input, shape, name=None, is_training=True):
    with tf.variable_scope(name):
        kernel = weight_variable(shape)
        c = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
        biases = bias_variable([shape[3]], name=name + "_biases")
        bias = tf.nn.bias_add(c, biases)
        with tf.device('/cpu:0'):
            conv = tf.nn.relu(
                tf.contrib.layers.batch_norm(bias, is_training=is_training, center=False, scope=name + "_bn"))
    return max_pool_2x2_argmax(conv, name=name + "_pool")


def conv_layer_with_bias(input, shape, name=None, is_training=True):
    with tf.variable_scope(name):
        kernel = weight_variable(shape)
        c = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')
        biases = bias_variable([shape[3]], name=name + "_biases")
        bias = tf.nn.bias_add(c, biases)
        conv = tf.contrib.layers.batch_norm(bias, is_training=is_training, center=False, scope=name + "_bn")

    return conv


def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl ** 2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)


def loss(logits, labels):
    with tf.name_scope('loss'):
        tf.cast(labels, tf.int32)
        # reshape logits to 1-D array of one hot vectors
        logits = tf.reshape(logits, (-1, NUM_CLASSES))
        epsilon = tf.constant(value=1e-10, name="epsilon")
        logits = tf.add(logits, epsilon)

        # one hot label
        label_flat = tf.reshape(labels, (-1, 1))
        with tf.device("/cpu:0"):
            # reshape labels to the same as logits
            labels = tf.reshape(tf.one_hot(label_flat, depth=NUM_CLASSES), (-1, NUM_CLASSES))
        # apply softmax on logits to extract label possibilities
        softmax = tf.nn.softmax(logits)
        # compute the cross entropy of logits vs labels
        cross_entropy = -tf.reduce_sum(labels * tf.log(softmax + epsilon), reduction_indices=[1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name="cross_entropy")
    return cross_entropy_mean


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
    # file names
    imageName = queue[0]
    labelName = queue[1]
    # read bytes into tensor
    imageVal = tf.read_file(imageName)
    labelVal = tf.read_file(labelName)
    # convert from png format
    imageB = tf.image.decode_png(imageVal)
    labelB = tf.image.decode_png(labelVal)
    # reshape to image/label shape
    image = tf.reshape(imageB, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    label = tf.reshape(labelB, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
    return image, label


def CamVid(images, labels, batch_size):
    img = ops.convert_to_tensor(images, dtype=dtypes.string)
    lbl = ops.convert_to_tensor(labels, dtype=dtypes.string)

    queue = tf.train.slice_input_producer([img, lbl], shuffle=True)
    image, label = reader(queue)
    reshaped = tf.cast(image, tf.float32)
    min_ex = int(0.4 * 367)
    return generate_batch(reshaped, label, min_ex, batch_size, shuffle=True)


def load_names(train=True):
    if train:
        train_dir = "CamVid/train/"
        annot = "CamVid/trainannot/"
        images = [train_dir + f for f in listdir(train_dir) if isfile(join(train_dir, f))]
        annotated = [annot + f for f in listdir(annot) if isfile(join(annot, f))]
    else:
        test_dir = "CamVid/test/"
        annot = "CamVid/testannot/"
        images = [test_dir + f for f in listdir(test_dir) if isfile(join(test_dir, f))]
        annotated = [annot + f for f in listdir(annot) if isfile(join(annot, f))]
    print("Done loading images...")
    assert (images[0][:10] == annotated[0][:10])
    return images, annotated


def setup(args, filter_size=32):
    x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="x")
    y_ = tf.placeholder(tf.uint8, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="y_")
    normal = tf.nn.lrn(x, depth_radius=5, bias=1.0, alpha=1e-4, beta=0.75, name="normalize")
    pool1, pool1_indices = conv_pool_layer_with_bias(normal, [7, 7, IMAGE_DEPTH, filter_size], name="pool1",
                                                     is_training=args.train)
    pool2, pool2_indices = conv_pool_layer_with_bias(pool1, [7, 7, filter_size, filter_size], name="pool2",
                                                     is_training=args.train)
    pool3, pool3_indices = conv_pool_layer_with_bias(pool2, [7, 7, filter_size, filter_size], name="pool3",
                                                     is_training=args.train)
    pool4, pool4_indices = conv_pool_layer_with_bias(pool3, [7, 7, filter_size, filter_size], name="pool4",
                                                     is_training=args.train)

    up4 = deconv_layer(pool4, [2, 2, filter_size, filter_size],
                       [BATCH_SIZE, IMAGE_HEIGHT // 8, IMAGE_WIDTH // 8, filter_size],
                       name="up4")
    de4 = conv_layer_with_bias(up4, [7, 7, filter_size, filter_size], name="de4", is_training=args.train)
    up3 = deconv_layer(de4, [2, 2, filter_size, filter_size],
                       [BATCH_SIZE, IMAGE_HEIGHT // 4, IMAGE_WIDTH // 4, filter_size],
                       name="up3")
    de3 = conv_layer_with_bias(up3, [7, 7, filter_size, filter_size], name="de3", is_training=args.train)
    up2 = deconv_layer(de3, [2, 2, filter_size, filter_size],
                       [BATCH_SIZE, IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2, filter_size],
                       name="up2")
    de2 = conv_layer_with_bias(up2, [7, 7, filter_size, filter_size], name="de2", is_training=args.train)
    up1 = deconv_layer(de2, [2, 2, filter_size, filter_size], [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, filter_size],
                       name="up1")
    de1 = conv_layer_with_bias(up1, [7, 7, filter_size, filter_size], name="de1", is_training=args.train)
    with tf.variable_scope('conv_classifier'):
        kernel = tf.get_variable('weights', [1, 1, filter_size, 3], initializer=msra_initializer(1, 64))
        conv = tf.nn.conv2d(de1, kernel, [1, 1, 1, 1], padding='SAME', name="conv")
        biases = bias_variable([3, 1], name="conv_biases")
        conv_classifier = tf.nn.bias_add(conv, biases, name="conv_classifier")
        # classifier = tf.nn.softmax(conv_classifier)
        mean_loss = loss(conv_classifier, y_)
    return x, y_, conv_classifier, mean_loss


def test(args, global_step):
    im, an = load_names(train=False)
    with tf.device('/cpu:0'):
        imgs, labels = CamVid(im, an, BATCH_SIZE)
    x, y_, conv_classifier, mean_loss = setup(args)
    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    variable_averages.apply(tf.trainable_variables())
    now = datetime.now()
    LOG_DIR = 'test/' + now.strftime("%Y%m%d-%H%M%S") + "/"
    with tf.Session() as sess:
        with tf.device('/cpu:0'):
            variable_averages = tf.train.ExponentialMovingAverage(
                0.9999)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
        ckpt = tf.train.get_checkpoint_state(args.check)
        if ckpt and ckpt.model_checkpoint_path:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("No checkpoint found! Please specify using the argument -ckpt")
            return
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(args.max):
            imageB, labelB = sess.run([imgs, labels])
            feed = {x: imageB, y_: labelB}
            pred = sess.run([conv_classifier], feed_dict=feed)

        coord.request_stop()
        coord.join(threads)


def train(args, global_step):
    im, an = load_names(True)

    x, y_, conv_classifier, mean_loss = setup(args)
    opt = tf.train.AdamOptimizer(args.lr).minimize(mean_loss, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
    var_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([opt, var_op]):
        train_op = tf.no_op(name="train")
    with tf.device('/cpu:0'):
        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        correct = tf.equal(
            tf.reshape(tf.argmax(tf.slice(conv_classifier, [0, 0, 0, 0], [1, -1, -1, -1]), 3), [-1, 360, 480, 1]),
            tf.cast(tf.slice(y_, [0, 0, 0, 0], [1, -1, -1, -1]), tf.int64))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.image('annotated', y_)
        tf.summary.image('correct', tf.cast(tf.reshape(correct, [-1, 360, 480, 1]), tf.uint8) * 255)
        tf.summary.image('input', x)
        tf.summary.image('output',
                         tf.cast(tf.reshape(tf.argmax(tf.slice(conv_classifier, [0, 0, 0, 0], [1, -1, -1, -1]), 3),
                                            [-1, 360, 480, 1]),
                                 tf.uint8) * 7)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('loss', mean_loss)

    LOG_DIR = 'train/' + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + str(args.max) + "/"

    print("Done setting up graph.")
    with tf.device('/cpu:0'):
        saver = tf.train.Saver(tf.global_variables())
        # load batch
        imgs, labels = CamVid(im, an, BATCH_SIZE)
        merged = tf.summary.merge_all()
    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(args.max):
            imageB, labelB = sess.run([imgs, labels])
            feed = {x: imageB, y_: labelB}
            _, l, a, summary = sess.run([train_op, mean_loss, accuracy, merged], feed_dict=feed)
            print(step, l, a)
            if step % 10 == 0:
                train_writer.add_summary(summary, step)
            if step > 50 and step % 100 == 0:
                # do testing
                print("Testing")

                with tf.device("/cpu:0"):
                    images, annotated = load_names(train=False)
                    inp, out = CamVid(images, annotated, batch_size=BATCH_SIZE)
                    testI, testL = sess.run([inp, out])
                print("Done processing images")
                feed = {x: testI, y_: testL}
                _, lo, acc = sess.run([train_op, mean_loss, accuracy], feed_dict=feed)
                print(lo, acc)
            if step % 500 == 0 or (step + 1) == args.max:
                saver.save(sess, join(LOG_DIR, 'model.ckpt'), global_step=step)
        coord.request_stop()
        coord.join(threads)


def main(args):
    global_step = tf.Variable(0, trainable=False)
    if args.train:
        train(args, global_step)
    else:
        test(args, global_step)


MAX_STEPS = 1000
NUM_CLASSES = 32
LEARNING_RATE = 1e-4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegNet in Tensorflow.")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("-lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("-max", type=int, default=MAX_STEPS, help="Max number of steps to take")
    parser.add_argument("-check", type=str, default=".", help="Checkpoint directory.")
    args = parser.parse_args()
    main(args)
