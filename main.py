import tensorflow as tf
import numpy as np
import pylab

#%%
size = 100

inputs = tf.placeholder(tf.float32, [None, size, size, 1], name="image")
lifegame = tf.placeholder(tf.int8, [1, size, size, 1], name="lifegame")

with tf.name_scope("count"):
    flt = tf.initializers.constant(np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
    ]))
    float_lifegame = tf.cast(lifegame, tf.float32)
    cnt = tf.contrib.slim.conv2d(float_lifegame, 1, 3, padding="same", activation_fn=None, weights_initializer=flt)
    input_cnt = tf.contrib.slim.conv2d(inputs * float_lifegame, 1, 3, padding="same", activation_fn=None, weights_initializer=flt)

with tf.name_scope("next_generation"):
    keep = tf.equal(lifegame, 1) & tf.greater_equal(cnt, 2) & tf.less_equal(cnt, 3)
    born = tf.equal(lifegame, 0) & tf.equal(cnt, 3)
    next_generation = keep | born

with tf.name_scope("image_mix"):
    target = keep # keep, born, next_generation
    mix_image = tf.cast(target, tf.float32) * input_cnt / tf.maximum(cnt, 1)
    mix_image = tf.cast(tf.logical_not(target), tf.float32) * inputs + mix_image

#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())

tf.summary.FileWriter("lglogs", sess.graph)

#%%
lg = np.random.randint(0, 100, (1, size, size, 1)) < 30
lg = lg.astype(np.int8)
img = np.random.randint(0, 255, (3, size, size, 1)).astype(np.float32)

with tf.device("/device:GPU:0"):
    for epoch in range(100):
        pylab.title("epoch : {}".format(epoch))
        pylab.imshow(np.squeeze(img.astype(np.uint8)).transpose(1, 2, 0))
        pylab.show()
        lg, img = sess.run([next_generation, mix_image], feed_dict={lifegame: lg.astype(np.int8), inputs: img})
