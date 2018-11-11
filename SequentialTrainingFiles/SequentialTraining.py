from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import time
import functools
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

# Main TFGAN library.
tfgan = tf.contrib.gan

# TFGAN MNIST examples from `tensorflow/models`.
from mnist import data_provider
from mnist import util

# TF-Slim data provider.
from datasets import download_and_convert_mnist

# Shortcuts for later.
queues = tf.contrib.slim.queues
layers = tf.contrib.layers
ds = tf.contrib.distributions
framework = tf.contrib.framework

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

### ### From Striner
def generate_grid(gan_model, latent_units):
    print("Generating grid images.\n")
    with tf.variable_scope(gan_model.generator_scope, reuse=True):
        with tf.name_scope('GeneratedImages/'):
            print("Entered scope\n")
            w = 10 #tf.flags.FLAGS.grid_size # We just want a 10x10 grid.
            print ("Set w to {}".format(w))
            n = w * w
            print ("Set n to {}".format(n))
            # Sample from the latent space
            print("Asking for random")
            rnd = tf.random_normal(shape=(n, latent_units), mean=0.,
                                   stddev=1., dtype=tf.float32, name='generated_rnd')
            print ("Got a random\n")
            # Generate images
            img = gan_model.generator_fn(rnd)
            print ("Ran generator function")
            # Reshape images into a grid
            iw = 28
            c = 1
            img = tf.reshape(img[:w * w], (w, w, iw, iw, c))
            img = tf.transpose(img, (0, 2, 1, 3, 4))
            img = tf.reshape(img, (1, iw * w, iw * w, c))
            # Rescale and clip to [0,1]
            img = tf.clip_by_value((img + 1.) / 2., 0., 1.)
            # Add summary
            tf.summary.image('generated-images', img)
### END NEW

######################### Copied TF libraries
### Direct copy from https://github.com/tensorflow/models/blob/master/research/gan/mnist/data_provider.py since the package excludes it now
# def provide_data(split_name, batch_size, dataset_dir, num_readers=1,
#                  num_threads=1):
#   """Provides batches of MNIST digits.
#   Args:
#     split_name: Either 'train' or 'test'.
#     batch_size: The number of images in each batch.
#     dataset_dir: The directory where the MNIST data can be found.
#     num_readers: Number of dataset readers.
#     num_threads: Number of prefetching threads.
#   Returns:
#     images: A `Tensor` of size [batch_size, 28, 28, 1]
#     one_hot_labels: A `Tensor` of size [batch_size, mnist.NUM_CLASSES], where
#       each row has a single element set to one and the rest set to zeros.
#     num_samples: The number of total samples in the dataset.
#   Raises:
#     ValueError: If `split_name` is not either 'train' or 'test'.
#   """
#   dataset = datasets.get_dataset('mnist', split_name, dataset_dir=dataset_dir)
#   provider = slim.dataset_data_provider.DatasetDataProvider(
#       dataset,
#       num_readers=num_readers,
#       common_queue_capacity=2 * batch_size,
#       common_queue_min=batch_size,
#       shuffle=(split_name == 'train'))
#   [image, label] = provider.get(['image', 'label'])

#   # Preprocess the images.
#   image = (tf.to_float(image) - 128.0) / 128.0

#   # Creates a QueueRunner for the pre-fetching operation.
#   images, labels = tf.train.batch(
#       [image, label],
#       batch_size=batch_size,
#       num_threads=num_threads,
#       capacity=5 * batch_size)

#   one_hot_labels = tf.one_hot(labels, dataset.num_classes)
#   return images, one_hot_labels, dataset.num_samples

############### And these two are from
# https://github.com/tensorflow/models/blob/master/research/gan/mnist/util.py





###################################




def visualize_training_generator(train_step_num, start_time, data_np):
    """Visualize generator outputs during training.

    Args:
        train_step_num: The training step number. A python integer.
        start_time: Time when training started. The output of `time.time()`. A
            python float.
        data: Data to plot. A numpy array, most likely from an evaluated TensorFlow
            tensor.
    """
    print('Training step: %i' % train_step_num)
    time_since_start = (time.time() - start_time) / 60.0
    print('Time since start: %f m' % time_since_start)
    print('Steps per min: %f' % (train_step_num / time_since_start))
    plt.axis('off')
    plt.imshow(np.squeeze(data_np), cmap='gray')

def visualize_digits(tensor_to_visualize):
    """Visualize an image once. Used to visualize generator before training.

    Args:
        tensor_to_visualize: An image tensor to visualize. A python Tensor.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with queues.QueueRunners(sess):
            images_np = sess.run(tensor_to_visualize)
    plt.axis('off')
    plt.imshow(np.squeeze(images_np), cmap='gray')

def evaluate_tfgan_loss(gan_loss, name=None):
    """Evaluate GAN losses. Used to check that the graph is correct.

    Args:
        gan_loss: A GANLoss tuple.
        name: Optional. If present, append to debug output.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with queues.QueueRunners(sess):
            gen_loss_np = sess.run(gan_loss.generator_loss)
            dis_loss_np = sess.run(gan_loss.discriminator_loss)
    if name:
        print('%s generator loss: %f' % (name, gen_loss_np))
        print('%s discriminator loss: %f' % (name, dis_loss_np))
    else:
        print('Generator loss: %f' % gen_loss_np)
        print('Discriminator loss: %f' % dis_loss_np)

MNIST_DATA_DIR = '/tmp/mnist-data'

if not tf.gfile.Exists(MNIST_DATA_DIR):
    tf.gfile.MakeDirs(MNIST_DATA_DIR)

download_and_convert_mnist.run(MNIST_DATA_DIR)

tf.reset_default_graph()

# Define our input pipeline. Pin it to the CPU so that the GPU can be reserved
# for forward and backwards propogation.
batch_size = 64
max_iters = 2501
with tf.device('/cpu:0'):
    real_images, _, _ = data_provider.provide_data(
        'train', batch_size, MNIST_DATA_DIR)

# Sanity check that we're getting images.
check_real_digits = tfgan.eval.image_reshaper(
    real_images[:20,...], num_cols=10)

visualize_digits(check_real_digits)
#plt.show()
#uncomment to see original mnist digits


def generator_fn(noise, weight_decay=2.5e-5, is_training=True):
    """Simple generator to produce MNIST images.

    Args:
        noise: A single Tensor representing noise.
        weight_decay: The value of the l2 weight decay.
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch
            norm uses the exponential moving average collected from population
            statistics.

    Returns:
        A generated image in the range [-1, 1].
    """
    with framework.arg_scope(
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(weight_decay)), \
         framework.arg_scope([layers.batch_norm], is_training=is_training,
                             zero_debias_moving_mean=True):
        net = layers.fully_connected(noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 256)
        net = tf.reshape(net, [-1, 7, 7, 256])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)

        return net


def discriminator_fn(img, unused_conditioning, weight_decay=2.5e-5,
                     is_training=True):
    """Discriminator network on MNIST digits.

    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        unused_conditioning: The TFGAN API can help with conditional GANs, which
            would require extra `condition` information to both the generator and the
            discriminator. Since this example is not conditional, we do not use this
            argument.
        weight_decay: The L2 weight decay.
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch
            norm uses the exponential moving average collected from population
            statistics.

    Returns:
        Logits for the probability that the image is real.
    """
    with framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        with framework.arg_scope([layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(net, 1024, normalizer_fn=layers.batch_norm)
        return layers.linear(net, 1)

noise_dims = 64
gan_model = tfgan.gan_model(
    generator_fn,
    discriminator_fn,
    real_data=real_images,
    generator_inputs=tf.random_normal([batch_size, noise_dims]))

# Sanity check that generated images before training are garbage.
check_generated_digits = tfgan.eval.image_reshaper(
    gan_model.generated_data[:20,...], num_cols=10)
visualize_digits(check_generated_digits)
#plt.show()
#uncomment to show original original shitty generated digits


improved_wgan_loss = tfgan.gan_loss(
    gan_model,
    # We make the loss explicit for demonstration, even though the default is
    # Wasserstein loss.
    # generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    # discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    generator_loss_fn = tfgan.losses.modified_generator_loss,
    discriminator_loss_fn = tfgan.losses.modified_discriminator_loss,
    gradient_penalty_weight=1.0)

#generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
#discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
generator_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
discriminator_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001)
gan_train_ops = tfgan.gan_train_ops(
    gan_model,
    improved_wgan_loss,
    generator_optimizer,
    discriminator_optimizer,
    check_for_unused_update_ops = False)

num_images_to_eval = 500
MNIST_CLASSIFIER_FROZEN_GRAPH = './mnist/data/classify_mnist_graph_def.pb'

# For variables to load, use the same variable scope as in the train job.
with tf.variable_scope('Generator', reuse=True):
    eval_images = gan_model.generator_fn(
        tf.random_normal([num_images_to_eval, noise_dims]),
        is_training=False)


# Calculate Inception score.
eval_score = util.mnist_score(eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH)

# Calculate Frechet Inception distance.
with tf.device('/cpu:0'):
    real_images, _, _ = data_provider.provide_data(
        'train', num_images_to_eval, MNIST_DATA_DIR)
frechet_distance = util.mnist_frechet_distance(
    real_images, eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH)

# Reshape eval images for viewing.
generated_data_to_visualize = tfgan.eval.image_reshaper(
    eval_images[:20,...], num_cols=10)
generated_data_to_visualize_tensor=tf.convert_to_tensor(generated_data_to_visualize)

train_step_fn = tfgan.get_sequential_train_steps()

global_step = tf.train.get_or_create_global_step()
loss_values, mnist_scores, frechet_distances  = [], [], []

digits_np=x = tf.placeholder(tf.float32, shape=(1024, 1024))


print("\n\n\t\tRUN_BEGINS_HERE\n")
## GENERATE-GRID CALL JS
# generate_grid(gan_model, latent_units=noise_dims)

sum1 = tf.summary.image(name='digits', tensor=generated_data_to_visualize_tensor)
print("Printing type of sum1")
print(type(sum1))
with tf.train.SingularMonitoredSession() as sess:
    writer=tf.summary.FileWriter('/tmp/mnist_demo/10')
#     print(type(sum1))
#     writer.add_summary(sum1)
    writer.add_graph(sess.graph)
    start_time = time.time()
    for i in xrange(max_iters):
        cur_loss, _ = train_step_fn(
            sess, gan_train_ops, global_step, train_step_kwargs={})
        loss_values.append((i, cur_loss))
        if i % 200 == 0:
            ## We'd like to drop in an image here!!!
            mnist_score, f_distance, digits_np,digits_tensor = sess.run(
                [eval_score, frechet_distance, generated_data_to_visualize,generated_data_to_visualize_tensor])
            mnist_scores.append((i, mnist_score))
            frechet_distances.append((i, f_distance))
            print('Current loss: %f' % cur_loss)
            print('Current MNIST score: %f' % mnist_scores[-1][1])
            print('Current Frechet distance: %f' % frechet_distances[-1][1])
            visualize_training_generator(i, start_time, digits_np)
    plt.show()
