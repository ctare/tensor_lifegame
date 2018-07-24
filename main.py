import tensorflow as tf
import numpy as np
import pylab

#%%
size = 50

inputs = tf.placeholder(tf.int8, [None, size, size, 1])

flt = tf.initializers.constant([
[1, 1, 1],
[1, 0, 1],
[1, 1, 1],
])
cnt = tf.contrib.slim.conv2d(tf.cast(inputs, tf.float32), 1, 3, padding="same", activation_fn=None, weights_initializer=flt)
outputs = (tf.equal(inputs, 1) & tf.greater_equal(cnt, 2) & tf.less_equal(cnt, 3)) | (tf.equal(inputs, 0) & tf.equal(cnt, 3))

#%%
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#%%
img = np.random.randint(0, 100, (1, size, size, 1)) < 30
img = img.astype(np.int8)

with tf.device("/device:GPU:0"):
    for _ in range(1000):
        pylab.imshow(np.squeeze(img))
        pylab.show()
        img = sess.run(outputs, feed_dict={inputs: img})
