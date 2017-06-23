"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
"""
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import utils
import sys
from datetime import datetime
import os
import scipy.misc
import scipy.io as sio
from PIL import Image

slim = tf.contrib.slim


def osvos_arg_scope(weight_decay=0.0002):
    """Defines the OSVOS arg scope.
    Args:
    weight_decay: The l2 regularization coefficient.
    Returns:
    An arg_scope.
    """
    with slim.arg_scope([slim.conv2d, slim.convolution2d_transpose, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.random_normal_initializer(stddev=0.001),
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        biases_initializer=tf.zeros_initializer(),
                        biases_regularizer=None,
                        padding='SAME') as arg_sc:
        return arg_sc


def crop_features(feature, out_size):
    """Crop the center of a feature map
    Args:
    feature: Feature map to crop
    out_size: Size of the output feature map
    Returns:
    Tensor that performs the cropping
    """
    up_size = tf.shape(feature)
    ini_w = tf.div(tf.subtract(up_size[1], out_size[1]), 2)
    ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2)
    slice_input = tf.slice(feature, (0, ini_w, ini_h, 0), (-1, out_size[1], out_size[2], -1))
    # slice_input = tf.slice(feature, (0, ini_w, ini_w, 0), (-1, out_size[1], out_size[2], -1))  # Caffe cropping way
    return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])


def osvos(inputs, num_classes, dropout_keep_prob=0.5, scope='osvos'):
    """Defines the OSVOS network
    Args:
    inputs: Tensorflow placeholder that contains the input image
    num_classes: number of classes in the classification network part
    dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
    scope: Scope name for the network
    Returns:
    net: Output Tensor of the network
    end_points: Dictionary with all Tensors of the network
    """
    im_size = tf.shape(inputs)

    with tf.variable_scope(scope, 'osvos', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        # Collect outputs of all intermediate layers.
        with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                            padding='SAME',
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.conv2d], trainable=False):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

            # LIAO: Fully connected layer for classification
            fc = slim.conv2d(net, 1024, [10, 10], padding='VALID', scope='fc6')
            fc = slim.dropout(fc, 0.5, is_training=True, scope='dropout6')
            fc7 = slim.conv2d(fc, 48, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='fc7')
            #fc7 = slim.dropout(fc7, 0.5, is_training=True, scope='dropout7')
            fc = slim.conv2d(fc7, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn=None, scope='fc8')
            fc = tf.squeeze(fc, [1, 2], name='fc8/squeezed')

        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, fc, fc7, end_points


def _train(dataset, valid_dataset, num_classes, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False, config=None, finetune=1,
           test_image_path=None, ckpt_name="osvos"):
    """Train OSVOS
    Args:
    dataset: Reference to a Dataset object instance
    initial_ckpt: Path to the checkpoint to initialize the network (May be parent network or pre-trained Imagenet)
    supervison: Level of the side outputs supervision: 1-Strong 2-Weak 3-No supervision
    learning_rate: Value for the learning rate. It can be a number or an instance to a learning rate object.
    logs_path: Path to store the checkpoints
    max_training_iters: Number of training iterations
    save_step: A checkpoint will be created every save_steps
    display_step: Information of the training will be displayed every display_steps
    global_step: Reference to a Variable that keeps track of the training steps
    iter_mean_grad: Number of gradient computations that are average before updating the weights
    batch_size: Size of the training batch
    momentum: Value of the momentum parameter for the Momentum optimizer
    resume_training: Boolean to try to restore from a previous checkpoint (True) or not (False)
    config: Reference to a Configuration object used in the creation of a Session
    finetune: Use to select the type of training, 0 for the parent network and 1 for finetunning
    test_image_path: If image path provided, every save_step the result of the network with this image is stored
    Returns:
    """
    model_name = os.path.join(logs_path, ckpt_name+".ckpt")
    if config is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        config.allow_soft_placement = True

    tf.logging.set_verbosity(tf.logging.INFO)

    # Prepare the input data
    input_image = tf.placeholder(tf.float32, [batch_size, None, None, 3])
    
    # LIAO: image label for classification part
    image_label = tf.placeholder(tf.float32, [batch_size, num_classes])

    # Create the network
    with slim.arg_scope(osvos_arg_scope()):
        net, fc, fc7, end_points = osvos(input_image, num_classes)

    # Define loss
    with tf.name_scope('losses'):

        fc = tf.nn.softmax(fc)
        classification_loss = tf.reduce_sum(tf.pow(fc-image_label,2)) / (2*batch_size)
        correct_pred = tf.equal(tf.argmax(fc, 1), tf.argmax(image_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        tf.summary.scalar('classification_loss', classification_loss)
        tf.summary.scalar('accuracy', accuracy)
            
        # LIAO: classification loss and l1 loss
        l2_loss = tf.add_n(tf.losses.get_regularization_losses())
        alpha = 0.025
        l1_loss = tf.reduce_sum(tf.abs(tf.subtract(tf.abs(fc7), tf.ones([fc7.shape[0],1,1,fc7.shape[3]])))) / batch_size
        total_loss = classification_loss + l2_loss + alpha * l1_loss
        tf.summary.scalar('l1_loss', l1_loss)
        tf.summary.scalar('l2_loss', l2_loss)
        tf.summary.scalar('total_loss', total_loss)

    # Define optimization method
    with tf.name_scope('optimization'):
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
        grads_and_vars = optimizer.compute_gradients(total_loss)
        with tf.name_scope('grad_accumulator'):
            grad_accumulator = {}
            for ind in range(0, len(grads_and_vars)):
                if grads_and_vars[ind][0] is not None:
                    grad_accumulator[ind] = tf.ConditionalAccumulator(grads_and_vars[ind][0].dtype)
        with tf.name_scope('apply_gradient'):
            grad_accumulator_ops = []
            for var_ind, grad_acc in grad_accumulator.iteritems():
                var_name = str(grads_and_vars[var_ind][1].name).split(':')[0]
                var_grad = grads_and_vars[var_ind][0]
                grad_accumulator_ops.append(grad_acc.apply_grad(var_grad,
                                                                local_step=global_step))
        with tf.name_scope('take_gradients'):
            mean_grads_and_vars = []
            for var_ind, grad_acc in grad_accumulator.iteritems():
                mean_grads_and_vars.append(
                    (grad_acc.take_grad(iter_mean_grad), grads_and_vars[var_ind][1]))
            apply_gradient_op = optimizer.apply_gradients(mean_grads_and_vars, global_step=global_step)
    # Log training info
    merged_summary_op = tf.summary.merge_all()

    # Initialize variables
    init = tf.global_variables_initializer()

    # Create objects to record timing and memory of the graph execution
    # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) # Option in the session options=run_options
    # run_metadata = tf.RunMetadata() # Option in the session run_metadata=run_metadata
    # summary_writer.add_run_metadata(run_metadata, 'step%d' % i)
    with tf.Session(config=config) as sess:
        print 'Init variable'
        sess.run(init)

        test_step = 100

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        valid_writer = tf.summary.FileWriter(os.path.join(logs_path,'valid'), graph=tf.get_default_graph())

        # Create saver to manage checkpoints
        saver = tf.train.Saver(max_to_keep=None)

        last_ckpt_path = tf.train.latest_checkpoint(logs_path)
        if last_ckpt_path is not None and resume_training:
            # Load last checkpoint
            print('Initializing from previous checkpoint...')
            saver.restore(sess, last_ckpt_path)
            step = global_step.eval() + 1
        else:
            print('Initializing from specified pre-trained model...')
            # init_weights(sess)
            var_list = []
            for var in tf.global_variables():
                # LIAO: ignore lack of fc
                if var.name.find('fc') != -1: continue 
                var_type = var.name.split('/')[-1]
                if 'weights' in var_type or 'bias' in var_type:
                    var_list.append(var)
            saver_res = tf.train.Saver(var_list=var_list)
            saver_res.restore(sess, initial_ckpt)
            step = 1
        #sess.run(interp_surgery(tf.global_variables()))
        print('Weights initialized')

        print 'Start training'
        while step < max_training_iters + 1:
            # Average the gradient
            for _ in range(0, iter_mean_grad):
                # LIAO: classification label one-hot encoding
                batch_image, _, batch_cls_label = dataset.next_batch(batch_size, 'train')
                for i in range(batch_size):
                   image = batch_image[i]
                   if type(image) is not np.ndarray:
                        image = np.array(Image.open(image), dtype=np.uint8)
                   image = image[:, :, ::-1]
                   image = np.subtract(image, np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
                   batch_image[i] = image 
                image = batch_image
                cls_label = slim.one_hot_encoding(batch_cls_label, num_classes).eval(session=sess) 

                # LIAO: classification label
                run_res = sess.run([total_loss, merged_summary_op, classification_loss, accuracy, l1_loss, l2_loss] + grad_accumulator_ops,
                                   feed_dict={input_image: image, image_label: cls_label})
                batch_loss = run_res[0]
                summary = run_res[1]
                cls_loss = run_res[2]
                acc = run_res[3]
                lloss = run_res[4]
                l2loss = run_res[5]

            # Apply the gradients
            sess.run(apply_gradient_op)  # Momentum updates here its statistics

            # Save summary reports
            summary_writer.add_summary(summary, step)

            # Display training status
            if step % display_step == 0:
                print >> sys.stderr, "{} Iter {}: Training Loss = {:.4f} l1 loss = {:.4f}, l2 loss = {:.4f}".format(datetime.now(), step, batch_loss, lloss, l2loss)
                print >> sys.stderr, "\t\tClassification Loss = {:.6f}, accuracy = {:.6f}".format(cls_loss, acc)

            # LIAO: validation
            if step % test_step == 0:
                valid_image, _, valid_cls_label = valid_dataset.next_batch(batch_size, 'train')
                for i in range(batch_size):
                   image = valid_image[i]
                   if type(image) is not np.ndarray:
                        image = np.array(Image.open(image), dtype=np.uint8)
                   image = image[:, :, ::-1]
                   image = np.subtract(image, np.array((104.00699, 116.66877, 122.67892), dtype=np.float32))
                   valid_image[i] = image 
                valid_cls_label = slim.one_hot_encoding(valid_cls_label, num_classes).eval(session=sess)  
                valid_res = sess.run([total_loss, merged_summary_op, classification_loss, accuracy, l1_loss, l2_loss],
                                    feed_dict={input_image: valid_image, image_label:valid_cls_label})
                valid_total_loss = valid_res[0]
                valid_summary = valid_res[1]
                valid_cls_loss = valid_res[2]
                valid_acc = valid_res[3]
                valid_l1loss = valid_res[4]
                valid_l2loss = valid_res[5]
                valid_writer.add_summary(valid_summary, step)
                print >> sys.stderr, "\n{} ***Test*** {}: Training Loss = {:.4f} l1 loss = {:.4f}, l2 loss = {:.4f} ".format(datetime.now(),
															step, valid_total_loss, valid_l1loss, valid_l2loss)
                print >> sys.stderr, "\t\tClassification Loss = {:.6f}, accuracy = {:.6f}".format(valid_cls_loss, valid_acc)
                print >> sys.stderr, "\t\t===== learning rate: {:.10f} =====\n".format(sess.run(learning_rate))
            
            # Save a checkpoint
            if step % save_step == 0:
                if test_image_path is not None:
                    curr_output = sess.run(img_summary, feed_dict={input_image: preprocess_img(test_image_path)})
                    summary_writer.add_summary(curr_output, step)
                save_path = saver.save(sess, model_name, global_step=global_step)
                print "Model saved in file: %s" % save_path

            step += 1

        if (step - 1) % save_step != 0:
            save_path = saver.save(sess, model_name, global_step=global_step)
            print "Model saved in file: %s" % save_path
        print('Finished training.')


def train_finetune(dataset, valid_dataset, num_classes, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step,
                   display_step, global_step, iter_mean_grad=1, batch_size=1, momentum=0.9, resume_training=False,
                   config=None, test_image_path=None, ckpt_name="osvos"):
    """Finetune OSVOS
    Args:
    See _train()
    Returns:
    """
    finetune = 1
    _train(dataset, valid_dataset, num_classes, initial_ckpt, supervison, learning_rate, logs_path, max_training_iters, save_step, display_step,
           global_step, iter_mean_grad, batch_size, momentum, resume_training, config, finetune, test_image_path,
           ckpt_name)
