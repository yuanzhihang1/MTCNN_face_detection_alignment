import tensorflow as tf

a=tf.Variable(tf.ones([3,3]))
b=tf.Variable(tf.ones([3,3]))
c=a*tf.cast(tf.equal(tf.reduce_mean(a),0),tf.float32)
d=tf.equal(tf.reduce_mean(a),1)
with tf.Session() as s:
    s.run(tf.initialize_all_variables())
    d=s.run(c)
    pass
