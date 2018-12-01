import numpy as np
import tensorflow as tf
import tensorflow.contrib.gan as tfgan
from tensorflow.contrib.gan.python.namedtuples import GANTrainSteps
from tensorflow.contrib.keras.api.keras.datasets import mnist
from tensorflow.contrib.layers.python.layers.regularizers import l2_regularizer
from tensorflow.contrib.learn import Experiment
from tensorflow.contrib.learn import RunConfig
from tensorflow.contrib.training import HParams
from tensorflow.python.estimator.estimator import Estimator


def generator_fn(latent):
    kwargs = {
        'kernel_regularizer': l2_regularizer(1e-6),
        'bias_regularizer': l2_regularizer(1e-6)}
    h = latent
    h = tf.layers.dense(h, 1024, name='latent_embedding_0', activation=tf.nn.leaky_relu, **kwargs)
    h = tf.layers.dense(h, 7 * 7 * 128, name='latent_embedding_1', activation=tf.nn.leaky_relu, **kwargs)
    h = tf.reshape(h, [-1, 7, 7, 128])
    h = tf.layers.conv2d_transpose(
        inputs=h,
        filters=64,
        kernel_size=[4, 4],
        strides=2,
        padding='same',
        name='generator_conv2d_1',
        activation=tf.nn.leaky_relu, **kwargs)
    h = tf.layers.conv2d_transpose(
        inputs=h,
        filters=32,
        kernel_size=[4, 4],
        strides=2,
        padding='same',
        name='generator_conv2d_2',
        activation=tf.nn.leaky_relu, **kwargs)
    h = tf.layers.conv2d(
        inputs=h,
        filters=1,
        kernel_size=[4, 4],
        padding='same',
        activation=tf.tanh,
        name='generator_conv2d_output', **kwargs)
    img = h
    return img


def discriminator_fn(img, _):
    kwargs = {
        'kernel_regularizer': l2_regularizer(1e-6),
        'bias_regularizer': l2_regularizer(1e-6)}
    h = img
    h = tf.layers.conv2d(
        h, 64, [4, 4], strides=2,
        padding='same', name='dis_conv2d_0', activation=tf.nn.leaky_relu, **kwargs)
    h = tf.layers.conv2d(
        h, 128, [4, 4], strides=2,
        padding='same', name='dis_conv2d_1', activation=tf.nn.leaky_relu, **kwargs)
    h = tf.layers.flatten(h)
    h = tf.layers.dense(h, 1024, name='dis_flat', activation=tf.nn.leaky_relu, **kwargs)
    h = tf.layers.dense(h, 1, name='dis_out', **kwargs)
    logits = h
    return logits


def make_input_fns():
    train, test = mnist.load_data()
    return make_input_fn(train), make_input_fn(test)


def make_input_fn(data, num_epochs=None):
    x, y = data
    x = (x.astype(np.float32) * 2. / 255.) - 1.
    x = np.expand_dims(x, axis=-1)
    fn = tf.estimator.inputs.numpy_input_fn(
        {'x': x}, y,
        batch_size=tf.flags.FLAGS.batch_size,
        num_epochs=num_epochs,
        shuffle=True
    )
    return fn


def generate_grid(gan_model, params):
    with tf.variable_scope(gan_model.generator_scope, reuse=True):
        with tf.name_scope('GeneratedImages/'):
            w = tf.flags.FLAGS.grid_size
            n = w * w
            # Sample from the latent space
            rnd = tf.random_normal(shape=(n, params.latent_units), mean=0.,
                                   stddev=1., dtype=tf.float32, name='generated_rnd')
            # Generate images
            img = gan_model.generator_fn(rnd)
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


def model_fn(features, labels, mode, params):
    if mode == tf.estimator.ModeKeys.PREDICT:
        raise NotImplementedError()
    else:
        # Pull images from input
        x = features['x']
        # Generate latent samples of same batch size as images
        n = tf.shape(x)[0]
        rnd = tf.random_normal(shape=(n, params.latent_units), mean=0., stddev=1., dtype=tf.float32)
        # Build GAN Model
        gan_model = tfgan.gan_model(
            generator_fn=generator_fn,
            discriminator_fn=discriminator_fn,
            real_data=x,
            generator_inputs=rnd)
        gan_loss = tfgan.gan_loss(
            gan_model,
            generator_loss_fn=tfgan.losses.modified_generator_loss,
            discriminator_loss_fn=tfgan.losses.modified_discriminator_loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
            generate_grid(gan_model, params)
            train_ops = tfgan.gan_train_ops(
                gan_model,
                gan_loss,
                generator_optimizer=tf.train.RMSPropOptimizer(params.gen_lr),
                discriminator_optimizer=tf.train.RMSPropOptimizer(params.dis_lr))
            gan_hooks = tfgan.get_sequential_train_hooks(GANTrainSteps(
                params.generator_steps, params.discriminator_steps
            ))(train_ops)
            return tf.estimator.EstimatorSpec(mode=mode, loss=gan_loss.discriminator_loss,
                                              train_op=train_ops.global_step_inc_op,
                                              training_hooks=gan_hooks)
        else:
            eval_metric_ops = {}
            return tf.estimator.EstimatorSpec(mode=mode, loss=gan_loss.discriminator_loss,
                                              eval_metric_ops=eval_metric_ops)


def experiment_fn(run_config, hparams):
    train_input_fn, eval_input_fn = make_input_fns()
    estimator = Estimator(
        model_fn=model_fn,
        config=run_config,
        params=hparams)
    experiment = Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn
    )

    return experiment


def main(_argv):
    model_dir = tf.flags.FLAGS.model_dir
    run_config = RunConfig(model_dir=model_dir)
    hparams = HParams(
        generator_steps=1,
        discriminator_steps=1,
        latent_units=100,
        dis_lr=1e-4,
        gen_lr=1e-3)
    tf.contrib.learn.learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule=tf.flags.FLAGS.schedule,
        hparams=hparams)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('model_dir', 'output/demo/gan_v1', 'Model directory')
    tf.flags.DEFINE_string('schedule', 'train_and_evaluate', 'Schedule')
    tf.flags.DEFINE_integer('batch_size', 128, 'Batch size')
    tf.flags.DEFINE_integer('grid_size', 10, 'grid_size')
    tf.flags.DEFINE_string('hparams', '', 'Hyperparameters')
    tf.app.run()
