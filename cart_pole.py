import tensorflow as tf
import numpy as np
import gym

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(1.0/shape[0], shape=shape)
  return tf.Variable(initial)

#Input value 
x   = tf.placeholder(tf.float32, [None, 4])

#Layer 1 (output)
W1    =   weight_variable([4,2])
b1    =   bias_variable([2])
y     =   tf.nn.softmax(tf.nn.relu(tf.matmul(x,W1)+b1))

#Output value (action)
out   =   tf.argmax(y, 1)[0]

env = gym.make('CartPole-v0')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while True:
        observation = env.reset()
        while True:
            env.render()
            action = sess.run(out, feed_dict={x:[observation]})
            observation, reward, done, info = env.step(action)
            if done:
                break
        
