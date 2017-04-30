import tensorflow as tf
import os
slim = tf.contrib.slim
import vgg
from single_label_dataset import Dataset

train_path = 'database/train_list'
valid_path = 'database/valid_list'
test_path = 'database/test_list'
dataset = Dataset(train_path, valid_path, 
                  test_path, '.', True)
ckpt_path = 'models/vgg_16.ckpt'
# hyperparameters
learning_rate = 0.001
momentum = 0.9
training_iters = 200000
batch_size = 32
test_step = 100
display_step = 10
# parameters
n_classes = 15
# placeholder
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.uint8, [None, n_classes])
# model
pred, end_points = vgg.vgg_16(x, n_classes)
# cost & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(cost)
# test
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuray = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# initialization
init = tf.initialize_all_variables()
variables_to_restore = slim.get_variables_to_restore(exclude=['vgg_16/fc8'])
init_fn = slim.assign_from_checkpoint_fn(ckpt_path, 
  variables_to_restore, ignore_missing_vars=True )

with tf.device('/gpu:0'):
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  with tf.Session(config=config) as sess:
    sess.run(init)  # default initialization
    init_fn(sess)   # load pre-train weights
    step = 1
    # training
    while step * batch_size < training_iters:
      batch_x, batch_y = dataset.next_batch(batch_size, 'train')
      batch_y = slim.one_hot_encoding(batch_y, n_classes).eval(session=sess)
      sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
      if step % display_step == 0:
        acc = sess.run(accuray, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print 'Iter ' + str(step*batch_size) + ': loss = {:.6f}'.format(loss) + '  accuracy = {:.5f}'.format(acc)
      if step % test_step == 0:
        batch_vx, batch_vy = dataset.next_batch(batch_size, 'valid')
        batch_vy = slim.one_hot_encoding(batch_vy, n_classes).eval(session=sess)
        acc = sess.run(accuray, feed_dict={x: batch_vx, y: batch_vy})
        loss = sess.run(cost, feed_dict={x: batch_vx, y: batch_vy})
        print 'Test ' + str(step*batch_size) + ': loss = {:.6f}'.format(loss) + '  accuracy = {:.5f}'.format(acc)
      step += 1
    print 'start save model...'
    saver = tf.train.Saver()
    save_path = saver.save(sess, "models/vgg_bbt.ckpt", global_step=step*batch_size)
    print ('model saved to ', save_path )
