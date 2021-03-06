#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import os
import json
import glob
import random
import collections
import math
import time

# # These are part of the infrastructure needed to run on Colab
DRIVE_BASE_DIR = '/home/js/group-proj/'
HANDBAG_DIR = DRIVE_BASE_DIR + 'handbags/500_bags'
FACADE_DIR = DRIVE_BASE_DIR + 'facades/train'

CROP_SIZE = 128

ARGSTRING = []

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
parser.add_argument("--output_dir", help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--dataset", choices=['handbags', 'facades'], default='handbags',
                    help="Choose dataset to use.")

parser.add_argument("--loss_fn", choices=['wgan', 'mod', 'minimax', 'pix2pix'],
                    default='pix2pix',
                    help="Loss function to use for generator and discriminator.")

parser.add_argument("--max_steps", type=int, default="5000",
                    help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, default="20",
                    help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=10, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=140, help="scale images to this size before cropping to CROPSIZE^2")
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

if len(ARGSTRING) > 0:
    a = parser.parse_args(ARGSTRING)
else:
    a = parser.parse_args()

if a.output_dir == "" or a.output_dir is None:
    a.output_dir = ("{}-s{}-e{}-batch{}-lr{}-L1wt{}-seed{}-{}"
                    .format(a.dataset, a.max_steps, a.max_epochs, a.batch_size,
                            a.lr, a.l1_weight, a.seed, a.loss_fn))
    i = 1
    stub = a.output_dir
    while(os.path.exists(a.output_dir)):
        a.output_dir = "{}.{}".format(stub, str(i).zfill(4))
        i = i + 1
    
if a.dataset == 'handbags':
    a.input_dir = HANDBAG_DIR
elif a.dataset == 'facades':
    a.input_dir = FACADE_DIR
else:
    raise ValueError("Unknown dataset.")

Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, predict_real, predict_fake, discrim_loss, discrim_grads_and_vars, gen_loss_GAN, gen_loss_L1, gen_grads_and_vars, train")


def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def discrim_conv(batch_input, out_channels, stride):
    padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    return tf.layers.conv2d(padded_input, out_channels, kernel_size=4, strides=(stride, stride), padding="valid", kernel_initializer=tf.random_normal_initializer(0, 0.02))


def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=True, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))


def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg

    if len(input_paths) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # if the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in input_paths):
        input_paths = sorted(input_paths, key=lambda path: int(get_name(path)))
    else:
        input_paths = sorted(input_paths)

    with tf.name_scope("load_images"):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=True)
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = decode(contents)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        with tf.control_dependencies([assertion]):
            raw_input = tf.identity(raw_input)

        raw_input.set_shape([None, None, 3])

        # break apart image pair and move to range [-1, 1]
        width = tf.shape(raw_input)[1] # [height, width, channels]
        left_images = preprocess(raw_input[:,:width//2,:])
        right_images = preprocess(raw_input[:,width//2:,:])

    if a.dataset == 'handbags':
        inputs, targets = [left_images, right_images]
    elif a.dataset == 'facades':
        inputs, targets = [right_images, left_images]
    else:
        raise ValueError("Unrecognized data set requested.")

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize_images(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)

        offset = tf.cast(tf.floor(tf.random_uniform([2], 0, a.scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
        if a.scale_size > CROP_SIZE:
            r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
        elif a.scale_size < CROP_SIZE:
            raise Exception("scale size cannot be less than crop size")
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = transform(targets)

    paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return Examples(
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(input_paths),
        steps_per_epoch=steps_per_epoch,
    )


def create_generator(generator_inputs, generator_outputs_channels):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    # or [batch, 128, 128, in_channels] => [batch, 64, 64, ngf], or whatever!
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)

    # The following automatically accommodates images of arbitrary size
    # provided the image dimension is a power-of-2.
    encoder_layer_specs = []
    for i in range (1, int(math.log(CROP_SIZE, 2))):
        v = a.ngf * (min(2**i, 8))
        encoder_layer_specs.append(v)
        
#     layer_specs = [
#         a.ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
#         a.ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
#         a.ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
#         a.ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
#         a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
#         a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]

#         ### We shrunk the image size, so have to drop a layer!!
#         ### Above sizes are off by a factor of 2!
# #        a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
#     ]
#     assert(layer_specs == encoder_layer_specs)

    for out_channels in encoder_layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved)
            layers.append(output)

    decoder_layer_specs = []
    for i in range (len(encoder_layer_specs), 0, -1):
        x = 0.0
        if i > 4: x = 0.5
        v = (a.ngf * (min(2**(i-1), 8)), x)
        decoder_layer_specs.append(v)
            
#     layer_specs = [
#         ### and the reverse of the above: drop a layer coming back out
# #        (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
#         (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
#         (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
#         (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
#         (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
#         (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
#         (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
#     ]
#     assert(layer_specs == decoder_layer_specs)
    
    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(decoder_layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                for i in range(0, len(layers)):
                    print("\t{}:\n{}\n"
                          .format(i, layers[i]))
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, CROP_SIZE/2, CROP_SIZE/2, ngf * 2] =>
    #            [batch, CROP_SIZE, CROP_SIZE, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]



######################### LOSS FUNCTIONS #######################################

EPS = 1e-12

## pix2pix losses are a functionalization of the original Hesse code.
def pix2pix_disc_loss(predict_real, predict_fake):
    with tf.name_scope("pix2pix_discriminator_loss"):
        # minimizing -tf.log will try to get inputs to 1
        # predict_real => 1
        # predict_fake => 0
        discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS)
                                        + tf.log(1 - predict_fake + EPS)))
    return discrim_loss


def pix2pix_gen_GAN_loss(predict_fake):
    with tf.name_scope("pix2pix_generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
    return gen_loss_GAN


## The following loss functions (wasserstein_discriminator_loss, wasserstein_generator_loss,
# minimax_discriminator_loss, minimax_generator_loss, modified_generator_loss)
# adapted from Shor. Original code at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/losses/python/losses_impl.py
# That code originally Copyright 2017 The TensorFlow Authors and licensed under the Apache License.
# See http://www.apache.org/licenses/LICENSE-2.0.

def wasserstein_discriminator_loss(
        discriminator_real_outputs,
        discriminator_gen_outputs):
    # For this experiment we weight real/generated losses equally
    real_weights = 1.0
    generated_weights = 1.0
    loss_collection = tf.GraphKeys.LOSSES
    reduction = tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    
    with tf.name_scope(None, 'discriminator_wasserstein_loss', (
            discriminator_real_outputs, discriminator_gen_outputs, real_weights,
            generated_weights)) as scope:
        discriminator_real_outputs = tf.to_float(discriminator_real_outputs)
        discriminator_gen_outputs = tf.to_float(discriminator_gen_outputs)
        discriminator_real_outputs.shape.assert_is_compatible_with(
            discriminator_gen_outputs.shape)
        
        loss_on_generated = tf.losses.compute_weighted_loss(
            discriminator_gen_outputs, generated_weights, scope,
            loss_collection=None, reduction=reduction)
        loss_on_real = tf.losses.compute_weighted_loss(
            discriminator_real_outputs, real_weights, scope, loss_collection=None,
            reduction=reduction)
        loss = loss_on_generated - loss_on_real
        tf.losses.add_loss(loss, loss_collection)
        
        tf.summary.scalar('discriminator_gen_wass_loss', loss_on_generated)
        tf.summary.scalar('discriminator_real_wass_loss', loss_on_real)
        tf.summary.scalar('discriminator_wass_loss', loss)

    return loss

def wasserstein_generator_loss(discriminator_gen_outputs):
    weights = 1.0
    loss_collection = tf.GraphKeys.LOSSES
    reduction = tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
  
    with tf.name_scope(None, 'generator_wasserstein_loss', (
            discriminator_gen_outputs, weights)) as scope:
        discriminator_gen_outputs = tf.to_float(discriminator_gen_outputs)

        loss = - discriminator_gen_outputs
        loss = tf.losses.compute_weighted_loss(
            loss, weights, scope, loss_collection, reduction)

        tf.summary.scalar('generator_wass_loss', loss)

    return loss


def minimax_discriminator_loss(discriminator_real_outputs,
                               discriminator_gen_outputs,
                               modified = False):
    label_smoothing = 0.25
    real_weights = 1.0
    generated_weights = 1.0
    loss_collection = tf.GraphKeys.LOSSES
    reduction = tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
    scope_name = 'discriminator_minimax_loss'
    if modified:
        scope_name = 'discriminator_modified_loss'
    with tf.name_scope(None, scope_name, (
            discriminator_real_outputs, discriminator_gen_outputs, real_weights,
            generated_weights, label_smoothing)) as scope:

        # -log((1 - label_smoothing) - sigmoid(D(x)))
        loss_on_real = tf.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_real_outputs),
            discriminator_real_outputs, real_weights, label_smoothing, scope,
            loss_collection=None, reduction=reduction)
        # -log(- sigmoid(D(G(x))))
        loss_on_generated = tf.losses.sigmoid_cross_entropy(
            tf.zeros_like(discriminator_gen_outputs),
            discriminator_gen_outputs, generated_weights, scope=scope,
            loss_collection=None, reduction=reduction)

        loss = loss_on_real + loss_on_generated
        tf.losses.add_loss(loss, loss_collection)

        tf.summary.scalar('discriminator_gen_minimax_loss', loss_on_generated)
        tf.summary.scalar('discriminator_real_minimax_loss', loss_on_real)
        tf.summary.scalar('discriminator_minimax_loss', loss)

    return loss

def minimax_generator_loss(discriminator_gen_outputs):
    with tf.name_scope(None, 'generator_minimax_loss') as scope:
        loss = - minimax_discriminator_loss(
            tf.ones_like(discriminator_gen_outputs),
            discriminator_gen_outputs)
        tf.summary.scalar('generator_minimax_loss', loss)

    return loss


def modified_discriminator_loss(discriminator_real_outputs,
                                discriminator_gen_outputs):
    return minimax_discriminator_loss(discriminator_real_outputs,
                                      discriminator_gen_outputs,
                                      modified = True)


def modified_generator_loss(discriminator_gen_outputs):
    label_smoothing = 0.0
    weights = 1.0
    loss_collection = tf.GraphKeys.LOSSES
    reduction = tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS

    with tf.name_scope(None, 'generator_modified_loss',
                        [discriminator_gen_outputs]) as scope:
        loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(discriminator_gen_outputs),
                                               discriminator_gen_outputs,
                                               weights,
                                               label_smoothing,
                                               scope,
                                               loss_collection,
                                               reduction)
        tf.summary.scalar('generator_modified_loss', loss)
    return loss

  ## Assign appropriate loss function to variable
DISCRIM_LOSS = None
GEN_LOSS = None
if (a.loss_fn == 'wgan'):
    DISCRIM_LOSS = wasserstein_discriminator_loss
    GEN_LOSS = wasserstein_generator_loss
elif (a.loss_fn == 'mod'):
    DISCRIM_LOSS = modified_discriminator_loss
    GEN_LOSS = modified_generator_loss
elif (a.loss_fn == 'minimax'):
    DISCRIM_LOSS = minimax_discriminator_loss
    GEN_LOSS = minimax_generator_loss
elif (a.loss_fn == 'pix2pix'):
    DISCRIM_LOSS = pix2pix_disc_loss
    GEN_LOSS = pix2pix_gen_GAN_loss
else:
    raise ValueError("Unrecognized loss function requested.")


################################################################################



def create_model(inputs, targets):
    def create_discriminator(discrim_inputs, discrim_targets):
        n_layers = 3
        layers = []

        # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
        input = tf.concat([discrim_inputs, discrim_targets], axis=3)

        # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
        with tf.variable_scope("layer_1"):
            convolved = discrim_conv(input, a.ndf, stride=2)
            rectified = lrelu(convolved, 0.2)
            layers.append(rectified)

        # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
        # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
        # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % (len(layers) + 1)):
                out_channels = a.ndf * min(2**(i+1), 8)
                stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                convolved = discrim_conv(layers[-1], out_channels, stride=stride)
                normalized = batchnorm(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            convolved = discrim_conv(rectified, out_channels=1, stride=1)
            output = tf.sigmoid(convolved)
            layers.append(output)

        return layers[-1]

    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables
    with tf.name_scope("real_discriminator"):
        with tf.variable_scope("discriminator"):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_real = create_discriminator(inputs, targets)

    with tf.name_scope("fake_discriminator"):
        with tf.variable_scope("discriminator", reuse=True):
            # 2x [batch, height, width, channels] => [batch, 30, 30, 1]
            predict_fake = create_discriminator(inputs, outputs)

    discrim_loss = DISCRIM_LOSS(predict_real, predict_fake)
    gen_loss_GAN = GEN_LOSS(predict_fake)
    with tf.name_scope("generator_loss"):
        if (a.l1_weight == 0.0):
            # don't bother to compute what we plan to ignore
            gen_loss_L1 = tf.constant(0.0)
        else:
            gen_loss_L1 = tf.reduce_mean(tf.abs(targets - outputs))
        gen_loss = gen_loss_GAN * a.gan_weight + gen_loss_L1 * a.l1_weight

    with tf.name_scope("discriminator_train"):
        discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
        discrim_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
        discrim_grads_and_vars = discrim_optim.compute_gradients(discrim_loss, var_list=discrim_tvars)
        discrim_train = discrim_optim.apply_gradients(discrim_grads_and_vars)

    with tf.name_scope("generator_train"):
        with tf.control_dependencies([discrim_train]):
            gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
            gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
            gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])

    global_step = tf.train.get_or_create_global_step()
    incr_global_step = tf.assign(global_step, global_step+1)

    return Model(
        predict_real=predict_real,
        predict_fake=predict_fake,
        discrim_loss=ema.average(discrim_loss),
        discrim_grads_and_vars=discrim_grads_and_vars,
        gen_loss_GAN=ema.average(gen_loss_GAN),
        gen_loss_L1=ema.average(gen_loss_L1),
        gen_grads_and_vars=gen_grads_and_vars,
        outputs=outputs,
        train=tf.group(update_losses, incr_global_step, gen_train),
    )


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        description = { "inputs": "edge", "targets": "base", "outputs": "faked" }
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + description[kind] + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, step=False):
    index_path = os.path.join(a.output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>input</th><th>output</th><th>target</th></tr>")

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s</td>" % fileset["name"])

        for kind in ["inputs", "outputs", "targets"]:
            index.write("<td><img src='images/%s'></td>" % fileset[kind])

        index.write("</tr>")
    return index_path


def main():
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.set_random_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples()
    print("examples count = %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, examples.targets)

    inputs = examples.inputs
    targets = examples.targets
    outputs = model.outputs

    def convert(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    # summaries
    with tf.name_scope("inputs_summary"):
        tf.summary.image("inputs", converted_inputs)

    with tf.name_scope("targets_summary"):
        tf.summary.image("targets", converted_targets)

    with tf.name_scope("outputs_summary"):
        tf.summary.image("outputs", converted_outputs)

    with tf.name_scope("predict_real_summary"):
        tf.summary.image("predict_real", tf.image.convert_image_dtype(model.predict_real, dtype=tf.uint8))

    with tf.name_scope("predict_fake_summary"):
        tf.summary.image("predict_fake", tf.image.convert_image_dtype(model.predict_fake, dtype=tf.uint8))

    tf.summary.scalar("discriminator_loss", model.discrim_loss)
    tf.summary.scalar("generator_loss_GAN", model.gen_loss_GAN)
    tf.summary.scalar("generator_loss_L1", model.gen_loss_L1)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + "/values", var)

    for grad, var in model.discrim_grads_and_vars + model.gen_grads_and_vars:
        tf.summary.histogram(var.op.name + "/gradients", grad)

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("parameter_count =", sess.run(parameter_count))

        max_steps = 2**32
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps

        # training
        start = time.time()

        for step in range(max_steps):
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            options = None
            run_metadata = None
            if should(a.trace_freq):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            fetches = {
                "train": model.train,
                "global_step": sv.global_step,
            }

            if should(a.progress_freq):
                fetches["discrim_loss"] = model.discrim_loss
                fetches["gen_loss_GAN"] = model.gen_loss_GAN
                fetches["gen_loss_L1"] = model.gen_loss_L1

            if should(a.summary_freq):
                fetches["summary"] = sv.summary_op

            if should(a.display_freq):
                fetches["display"] = display_fetches

            results = sess.run(fetches, options=options, run_metadata=run_metadata)

            if should(a.summary_freq):
                print("recording summary")
                sv.summary_writer.add_summary(results["summary"], results["global_step"])

            if should(a.display_freq):
                print("saving display images")
                filesets = save_images(results["display"], step=results["global_step"])
                append_index(filesets, step=True)

            if should(a.trace_freq):
                print("recording trace")
                sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % results["global_step"])

            if should(a.progress_freq):
                train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                rate = (step + 1) * a.batch_size / (time.time() - start)
                remaining = (max_steps - step) * a.batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss_GAN", results["gen_loss_GAN"])
                print("gen_loss_L1", results["gen_loss_L1"])

            if should(a.save_freq):
                print("saving model")
                saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

            if sv.should_stop():
                break


main()
