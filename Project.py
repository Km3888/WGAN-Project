from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import time
import functools
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
import argparse
import os
import json
import glob
import random
import collections
import math
import time

import networks
from slim.nets import cyclegan
from slim.nets import pix2pix


# Main TFGAN library.
tfgan = tf.contrib.gan

parser = argparse.ArgumentParser()
if True:
    parser.add_argument("--input_dir",default='/Users/kellymarshall/PycharmProjects/WGAN_Project/pix2pix-tensorflow/tools/edges2handbags/train', help="path to folder containing images")#added default argument
    parser.add_argument("--output_dir",default='Project/output7', help="where to put output files")#Removed required and added default argument

    parser.add_argument("--max_steps", default=1000,type=int, help="number of training steps (0 to disable)")#added default argument
    parser.add_argument("--max_epochs", type=int, help="number of training epochs")#added default argument
    parser.add_argument("--summary_freq", type=int, default=1, help="update summaries every summary_freq steps")
    parser.add_argument("--progress_freq", type=int, default=1, help="display progress every progress_freq steps")
    parser.add_argument("--trace_freq", type=int, default=1, help="trace execution every trace_freq steps")
    parser.add_argument("--display_freq", type=int, default=1, help="write current training images every display_freq steps")
    parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

    parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
    parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
    parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
    parser.add_argument("--batch_size", type=int, default=2, help="number of images in batch")
    parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
    parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
    parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
    parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
    parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
    parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
    parser.set_defaults(flip=True)
    parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
    parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
    parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
    parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

    parser.add_argument("--n_critic", type=int, default=5, help="Number of training iterations for the discriminator for each generator iteration")

a = parser.parse_args()
EPS = 1e-12
CROP_SIZE = 256


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
Examples = collections.namedtuple("Examples", "paths, inputs, targets, count, steps_per_epoch")

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def generator(input_images):
  """Thin wrapper around CycleGAN generator to conform to the TFGAN API.
  Args:
    input_images: A batch of images to translate. Images should be normalized
      already. Shape is [batch, height, width, channels].
  Returns:
    Returns generated image batch.
  Raises:
    ValueError: If shape of last dimension (channels) is not defined.
  """
  input_images.shape.assert_has_rank(4)
  input_size = input_images.shape.as_list()
  channels = input_size[-1]
  if channels is None:
    raise ValueError(
        'Last dimension shape must be known but is None: %s' % input_size)
  with tf.contrib.framework.arg_scope(cyclegan.cyclegan_arg_scope()):
    output_images, _ = cyclegan.cyclegan_generator_resnet(input_images,
                                                          num_outputs=channels)
  return output_images

def discriminator(image_batch, unused_conditioning=None):
  """A thin wrapper around the Pix2Pix discriminator to conform to TFGAN API."""
  with tf.contrib.framework.arg_scope(pix2pix.pix2pix_arg_scope()):
    logits_4d, _ = pix2pix.pix2pix_discriminator(
        image_batch, num_filters=[64, 128, 256, 512])
    logits_4d.shape.assert_has_rank(4)
  # Output of logits is 4D. Reshape to 2D, for TFGAN.
  logits_2d = tf.contrib.layers.flatten(logits_4d)

  return logits_2d


def save_images(fetches, step=None):
    image_dir = os.path.join(a.output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        for kind in ["inputs", "outputs", "targets"]:
            filename = name + "-" + kind + ".png"
            if step is not None:
                filename = "%08d-%s" % (step, filename)
            fileset[kind] = filename
            out_path = os.path.join(image_dir, filename)
            contents = fetches[kind][i]
            with open(out_path, "wb") as f:
                f.write(contents)
        filesets.append(fileset)
    return filesets
def load_examples():
    if a.input_dir is None or not os.path.exists(a.input_dir):
        raise Exception("input_dir does not exist")

    input_paths = glob.glob(os.path.join(a.input_dir, "*.jpg"))
    decode = tf.image.decode_jpeg
    if len(input_paths) == 0:
        input_paths = glob.glob(os.path.join(a.input_dir, "*.png"))
        decode = tf.image.decode_png

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
        path_queue = tf.train.string_input_producer(input_paths, shuffle=True)#we're assuming that it's always training
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
        a_images = preprocess(raw_input[:,:width//2,:])
        b_images = preprocess(raw_input[:,width//2:,:])

    #if a.which_direction == "BtoA":
    inputs, targets = [b_images, a_images]#We're assuming BtoA
    '''elif a.which_direction == "AtoB":
        inputs, targets = [a_images, b_images]
    else:
        raise Exception("invalid direction")'''

    # synchronize seed for image operations so that we do the same operations to both
    # input and output images
    seed = random.randint(0, 2**31 - 1)
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

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

def main():
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)
    examples = load_examples()
    inputs=examples.inputs
    targets=examples.targets

    gan_model = tfgan.gan_model(
        generator_fn=networks.generator,
        discriminator_fn=networks.discriminator,
        real_data=inputs,
        generator_inputs=targets)

    outputs=gan_model.generated_data

    with tf.name_scope('losses'):
      gan_loss = tfgan.gan_loss(
          gan_model,
          generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
          discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss)

    with tf.name_scope('gan_train_ops'):
        generator_optimizer = tf.train.RMSPropOptimizer(0.001)
        discriminator_optimizer = tf.train.RMSPropOptimizer(0.0001)
        gan_train_ops = tfgan.gan_train_ops(
            gan_model,
            gan_loss,
            generator_optimizer,
            discriminator_optimizer)

    steps = tfgan.GANTrainSteps(
        generator_train_steps=1,
        discriminator_train_steps=a.n_critic)



    def convert(image):
        return tf.image.convert_image_dtype(image, dtype=tf.uint8, saturate=True)

    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)

    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets)

    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs)

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

    generator_loss,discriminator_loss=gan_loss
    tf.summary.scalar("discriminator_loss", discriminator_loss)
    tf.summary.scalar("generator_loss_GAN", generator_loss)

    train_step_fn = tfgan.get_sequential_train_steps(train_steps=steps)
    global_step = tf.train.get_or_create_global_step()


    logdir = a.output_dir if (a.trace_freq > 0 or a.summary_freq > 0) else None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)

    with sv.managed_session() as sess:
        if a.max_epochs is not None:
            max_steps = examples.steps_per_epoch * a.max_epochs
        if a.max_steps is not None:
            max_steps = a.max_steps


        start=time.time()
        for step in range(max_steps):
            print('step=',step)
            def should(freq):
                return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)

            options = None
            run_metadata = None
            if should(a.trace_freq):
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
            print('running sess1')
            curr_loss,glo_gang=train_step_fn(
                sess, gan_train_ops, global_step, train_step_kwargs={})
            print('ran sess1')
            fetches = {
            }

            if should(a.progress_freq):
                fetches["discrim_loss"],fetches['gen_loss'] = gan_loss

            if should(a.summary_freq):
                fetches["summary"] = sv.summary_op

            if should(a.display_freq):
                fetches["display"] = display_fetches

            # values in fetches are the variables that get computed by sess
            print('running sess2..')
            results = sess.run(fetches, options=options, run_metadata=run_metadata)
            print('ran sess2')
            if should(a.summary_freq):
                print("recording summary")
                sv.summary_writer.add_summary(results["summary"], glo_gang)

            if should(a.display_freq):
                print("saving display images")
                filesets = save_images(results["display"], step=step+1)
                append_index(filesets, step=True)

            if should(a.trace_freq):
                print("recording trace")
                sv.summary_writer.add_run_metadata(run_metadata, "step_%d" % step)

            if should(a.progress_freq):
                # global_step will have the correct step count if we resume from a checkpoint
                train_epoch = math.ceil(glo_gang / examples.steps_per_epoch)
                train_step = (glo_gang - 1) % examples.steps_per_epoch + 1
                rate = (step + 1) * a.batch_size / (time.time() - start)
                remaining = (max_steps - step) * a.batch_size / rate
                print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (
                train_epoch, train_step, rate, remaining / 60))
                print("discrim_loss", results["discrim_loss"])
                print("gen_loss", results["gen_loss"])

            if should(a.save_freq):
                print("saving model")
                saver.save(sess, os.path.join(a.output_dir, "model"), global_step=sv.global_step)

if __name__=='__main__':
    main()