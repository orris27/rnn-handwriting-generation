import tensorflow as tf
import numpy as np
import model as m
from config import *

from utils import DataLoader, draw_strokes_random_color

data_loader = DataLoader(args.batch_size, args.T, args.data_scale,
                         chars=args.chars, points_per_char=args.points_per_char)

print('number of batches:', data_loader.num_batches)
args.U = data_loader.max_U
args.c_dimension = len(data_loader.chars) + 1
args.action = 'train'

model = m.Model(args)
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    saver = tf.train.Saver(tf.all_variables())
    for e in range(args.num_epochs):
        print("epoch %d" % e)
        data_loader.reset_batch_pointer()
        for b in range(data_loader.num_batches):
            x, y, c_vec, c = data_loader.next_batch()
            feed_dict = {model.x: x, model.y: y}
            if b % 100 == 0:
                loss = sess.run(model.loss, feed_dict=feed_dict)
                print('batches %d, loss %g' % (b, loss))
            sess.run(model.train_op, feed_dict=feed_dict)
        saver.save(sess, 'save_%s/model.tfmodel' % args.mode, global_step=e)
