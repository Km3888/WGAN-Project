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

########## Global constants / hyperparameters and environment setup
tf.flags.DEFINE_string('output_dir', '/tmp/mnist_demo/16', 'Model directory')
tf.flags.DEFINE_string('mnist_data_dir', '/tmp/mnist-data', 'MNIST data dir')
tf.flags.DEFINE_string('frozen_graph', './mnist/data/classify_mnist_graph_def.pb',
                       'Weights for MNIST evaluation')
tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.flags.DEFINE_integer('num_cols', 10, 'Output image columns')
tf.flags.DEFINE_float('weight_decay', 2.5e-5, 'Weight decay')
tf.flags.DEFINE_integer('noise_dim', 64, 'Dimension of images/noise')
tf.flags.DEFINE_integer('num_imgs_to_eval', 500,
                        'Number of images to eval for monitoring.')
tf.flags.DEFINE_float('grad_penalty_wt', 1.0, 'Gradient penalty weight')
tf.flags.DEFINE_float('gen_alpha', .001, 'RMSProp learning rate for generator')
tf.flags.DEFINE_float('disc_alpha', .001, 'RMSProp learning rate for discriminator')
tf.flags.DEFINE_integer('nCritic', 5, 'Critic training passes per iteration')
tf.flags.DEFINE_integer('max_steps', 2501, 'Max iterations to train')
tf.flags.DEFINE_integer('report_interval', 200, 'How often to report rates')



def visualize_training_generator(train_step_num, start_time, data_np,name):
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
    np.savetxt(name,np.squeeze(data_np))
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

if not tf.gfile.Exists(tf.flags.FLAGS.mnist_data_dir):
    tf.gfile.MakeDirs(tf.flags.FLAGS.mnist_data_dir)

download_and_convert_mnist.run(tf.flags.FLAGS.mnist_data_dir)

tf.reset_default_graph()

# Define our input pipeline. Pin it to the CPU so that the GPU can be reserved
# for forward and backwards propogation.
with tf.device('/cpu:0'):
    real_images, _, _ = data_provider.provide_data(
        'train', tf.flags.FLAGS.batch_size, tf.flags.FLAGS.mnist_data_dir)

# Sanity check that we're getting images.
check_real_digits = tfgan.eval.image_reshaper(
    real_images[:20,...], num_cols=tf.flags.FLAGS.num_cols)

visualize_digits(check_real_digits)
#plt.show()
#uncomment to see original mnist digits


def generator_fn(noise,
                 weight_decay=tf.flags.FLAGS.weight_decay,
                 is_training=True):
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


def discriminator_fn(img, unused_conditioning,
                     weight_decay=tf.flags.FLAGS.weight_decay,
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

gan_model = tfgan.gan_model(
    generator_fn,
    discriminator_fn,
    real_data=real_images,
    generator_inputs=tf.random_normal([tf.flags.FLAGS.batch_size,
                                       tf.flags.FLAGS.noise_dim]))


# Sanity check that generated images before training are garbage.
check_generated_digits = tfgan.eval.image_reshaper(
    gan_model.generated_data[:20,...], num_cols=tf.flags.FLAGS.num_cols)
visualize_digits(check_generated_digits)
#plt.show()
#uncomment to show original original (poor) generated digits


gan_loss = tfgan.gan_loss(
    gan_model,
    # NOTE: Modify the following 2 lines to change loss function
    generator_loss_fn=tfgan.losses.modified_generator_loss,
    discriminator_loss_fn=tfgan.losses.modified_discriminator_loss,
    gradient_penalty_weight=tf.flags.FLAGS.grad_penalty_wt)

generator_optimizer = tf.train.RMSPropOptimizer(tf.flags.FLAGS.gen_alpha)
discriminator_optimizer = tf.train.RMSPropOptimizer(tf.flags.FLAGS.disc_alpha)
gan_train_ops = tfgan.gan_train_ops(
    gan_model,
    gan_loss,
    generator_optimizer,
    discriminator_optimizer)


# For variables to load, use the same variable scope as in the train job.
with tf.variable_scope('Generator', reuse=True):
    eval_images = gan_model.generator_fn(
        tf.random_normal([tf.flags.FLAGS.num_imgs_to_eval,
                          tf.flags.FLAGS.noise_dim]),
        is_training=False)


# Calculate Inception score.
eval_score = util.mnist_score(eval_images, tf.flags.FLAGS.frozen_graph)

# Calculate Frechet Inception distance.
with tf.device('/cpu:0'):
    real_images, _, _ = data_provider.provide_data('train',
        tf.flags.FLAGS.num_imgs_to_eval, tf.flags.FLAGS.mnist_data_dir)
frechet_distance = util.mnist_frechet_distance(
    real_images, eval_images, tf.flags.FLAGS.frozen_graph)

# Reshape eval images for viewing.
generated_data_to_visualize = tfgan.eval.image_reshaper(
    eval_images[:20,...], num_cols=tf.flags.FLAGS.num_cols)
generated_data_to_visualize_tensor=tf.convert_to_tensor(generated_data_to_visualize)

steps=tf.contrib.gan.GANTrainSteps(1, tf.flags.FLAGS.nCritic)
train_step_fn = tfgan.get_sequential_train_steps(train_steps=steps)

global_step = tf.train.get_or_create_global_step()
loss_values, mnist_scores, frechet_distances  = [], [], []

digits_np = tf.placeholder(tf.float32, shape=(1024, 1024))

sum1 = tf.summary.image(name='digits', tensor=generated_data_to_visualize_tensor)
with tf.train.SingularMonitoredSession() as sess:
    writer=tf.summary.FileWriter(tf.flags.FLAGS.output_dir)
    writer.add_graph(sess.graph)
    start_time = time.time()
    for i in xrange(tf.flags.FLAGS.max_steps):
        cur_loss, _ = train_step_fn(
            sess, gan_train_ops, global_step, train_step_kwargs={})
        loss_values.append((i, cur_loss))
        if i % 10 == 0:
            mnist_score, f_distance, digits_np = sess.run(
                [eval_score, frechet_distance, generated_data_to_visualize,])
            mnist_scores.append((i, mnist_score))
            frechet_distances.append((i, f_distance))
            print('i=',i)
            print('Current loss: %f' % cur_loss)
            print('Current MNIST score: %f' % mnist_scores[-1][1])
            print('Current Frechet distance: %f' % frechet_distances[-1][1])
            print('-'*8)
            if i % tf.flags.FLAGS.report_interval == 0:
                name = 'GAN_images_' + 'i' + str(i) + '_time' + str(time.time() - start_time)+'v1'
                visualize_training_generator(i, start_time, digits_np,name=name)

    np.savetxt('GANLosses',loss_values)


    print('training time:',time.time()-start_time)



    plt.show()
