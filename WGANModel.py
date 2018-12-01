import tensorflow as tf
import numpy as np

# NOTE: Our next revision will avoid duplicating files and instead implement
# argparse to set the loss function chosen.

tfgan=tf.contrib.gan
mnist=tf.keras.datasets.mnist

EXAMPLECOUNT = 64

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train[:EXAMPLECOUNT], axis=-1).astype(np.float32)
x_test = np.expand_dims(x_test[:EXAMPLECOUNT], axis=-1).astype(np.float32)

queues = tf.contrib.slim.queues
layers = tf.contrib.layers
ds = tf.contrib.distributions
framework = tf.contrib.framework
leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

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
        net = layers.conv2d(img, EXAMPLECOUNT, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        with framework.arg_scope([layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(net, 1024, normalizer_fn=layers.batch_norm)
        return layers.linear(net, 1)

    

# gen_inputs=tf.random_normal([60000, 64])
# it turns out using 60k examples eventually kills your memory on a desktop!
gen_inputs=tf.random_normal([64, 64])

gan_model=tfgan.gan_model(generator_fn=generator_fn,
                          discriminator_fn=discriminator_fn,
                          real_data=x_train,
                          generator_inputs=gen_inputs)

wgan_loss=tfgan.gan_loss(gan_model,
                         generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
                         discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
                         gradient_penalty_weight=1.0)
gan_train_ops = tfgan.gan_train_ops(
    gan_model,
    wgan_loss,
    tf.train.RMSPropOptimizer(0.00005),
    tf.train.RMSPropOptimizer(0.00005))

tfgan.gan_train(
    gan_train_ops,
    hooks=[tf.train.StepCounterHook(10), tf.train.StopAtStepHook(num_steps=50000)],
    logdir='logs')


# Following is WIP code. Not used at present.
# # # thank you https://stackoverflow.com/questions/45532365/is-there-any-tutorial-for-tf-train-sessionrunhook#46795160
# # class _MonitorHook(tf.train.SessionRunHook):
# #     """Make TensorBoard see things"""

# #     # def begin(self):
# #     #     with tf.name_scope(layer_name):
# #     #         with tf.name_scope('summaries'):
# #     #             weights = weight_variable([input_dim, output_dim])
# #     #             mean = tf.reduce_mean(weights)
# #     #             tf.summary.scalar('mean', mean)
# #     #             tf.summary.histogram('histogram', weights)

# #     def begin(self):
# #         self._step = -1
# #         self._start_time = time.time()

# #     def before_run(self, run_context):
# #         self._step += 1
# #         return tf.train.SessionRunArgs(loss)  # Asks for loss value.

# #     def after_run(self, run_context, run_values):
# #         if self._step % FLAGS.log_frequency == 0:
# #             current_time = time.time()
# #             duration = current_time - self._start_time
# #             self._start_time = current_time

# #             loss_value = run_values.results
# #             examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
# #             sec_per_batch = float(duration / FLAGS.log_frequency)

# #             format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
# #                           'sec/batch)')
# #             print (format_str % (datetime.now(), self._step, loss_value,
# #                                  examples_per_sec, sec_per_batch))
