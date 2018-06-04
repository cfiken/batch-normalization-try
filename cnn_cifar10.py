import tensorflow as tf
import numpy as np
from models.cnn import CNN
from data_loader.cifar10 import DataLoader
from utils import helper
from datetime import datetime



learning_rate = 0.0001
batch_size  =128
num_epoch = 100
num_print_step = 50
dense_dropout_rate = 0.4

myint = tf.int32
myfloat = tf.float32


class CNNRunner:

    def __init__(self):
        config = {}
        self.cnn = CNN(config)
        self.datasource = DataLoader()

    def run(self):
        self.cnn
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.cnn.y, logits=self.cnn.logits)
        loss_op = tf.reduce_mean(crossent)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss_op)
        correct = tf.equal(self.cnn.predicted_classes, self.cnn.y)
        acc_op = tf.reduce_mean(tf.cast(correct, myfloat))

        with tf.name_scope('train'):
            smr_loss = tf.summary.scalar('loss', loss_op)
            smr_acc = tf.summary.scalar('accuracy', acc_op)
            merged_summary = tf.summary.merge([smr_loss, smr_acc])

        with tf.name_scope('test'):
            test_smr_acc = tf.summary.scalar('accuracy', acc_op)

        now = datetime.now()
        logdir_base = 'logs/'
        logdir = logdir_base + now.strftime("%Y%m%d-%H%M%S") + "/"

        with tf.Session() as sess:
            writer = tf.summary.FileWriter(logdir, sess.graph)
            sess.run(tf.global_variables_initializer())
            
            global_step = 0
            for i in range(num_epoch):
                step_size = self.datasource.num_step(batch_size)
                for s in range(step_size):
                    data, labels = self.datasource.next_batch(batch_size)
                    fd = {
                        self.cnn.x: data,
                        self.cnn.y: labels,
                        self.cnn.is_training: True
                    }
                    loss, _, acc, smr = sess.run([loss_op, train_op, acc_op, merged_summary], feed_dict=fd)
                    if s % num_print_step == 0:
                        writer.add_summary(smr, global_step)
                        print('{} steps, train accuracy: {:.6f}, loss: {:.6f}'.format(global_step, acc, loss))
                        test_acc, test_smr = sess.run([acc_op, test_smr_acc], feed_dict={
                            self.cnn.x: self.datasource.test_data,
                            self.cnn.y: self.datasource.test_labels,
                            self.cnn.is_training: False
                        })
                        writer.add_summary(test_smr, global_step)
                    global_step += 1
                print('{} steps, test accuracy:  {:.6f}, loss: {:.6f} ({}/{} epochs)'.format(global_step, test_acc, loss, i, num_epoch))


if __name__ == '__main__':
    CNNRunner().run()

