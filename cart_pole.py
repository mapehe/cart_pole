import tensorflow as tf
import numpy as np
import gym


"""
    The functions weight_variable and bias_variable
    are used to initialize the net. The discount_rewards
    function is used to boost the rewards of actions that are
    beneficial in the long run.
"""

gamma = 0.99 #Discount constant

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(1.0/shape[0], shape=shape)
  return tf.Variable(initial)

def discount_rewards(r):
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, len(r))):
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

"""
    We build a standard three layer net with
    ReLU activations.
"""

#Input value 
x     =   tf.placeholder(tf.float32, [None, 4])

#Layer 1
W1    =   weight_variable([4,8])
b1    =   bias_variable([8])
h1    =   tf.nn.relu(tf.matmul(x,W1)+b1)

#Layer 2
W2    =   weight_variable([8,4])
b2    =   bias_variable([4])
h2    =   tf.nn.relu(tf.matmul(h1,W2)+b2)

#Layer 3 (output)
W3    =   weight_variable([4,2])
b3    =   bias_variable([2])
y     =   tf.nn.softmax(tf.nn.relu(tf.matmul(h2,W3)+b3))

y_      =   tf.placeholder(tf.uint8, [None])
yhot    =   tf.one_hot(y_, depth=2)
rewards =   tf.placeholder(tf.float32, [None])

"""
    The loss function simlifies to the sum of terms
    -r_t*log(y_t), where t indexes the game's frames, y_t
    is the probability of choosing the action our stochastic
    policy yielded at time t and r_t is the reward for
    that action.
"""

loss    =   tf.tensordot( rewards,
                          tf.reduce_sum(-yhot * tf.log(y), reduction_indices = [1]),
                          axes=1)

training_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

env = gym.make('CartPole-v0')

"""
    Run the game. Every time there is a game over, collect the data
    from the game and use it to update the network. The game is usually
    solved in a few minutes.
"""

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while True:
        observation = env.reset()
        o_s = [observation]
        a_s = []
        r_s = []
        while True:
            env.render()
            prob_0 = sess.run(y, feed_dict={x:[observation]})[0][0]
            if np.random.random() < prob_0:
                action = 0
            else:
                action = 1
            a_s += [action]
            observation, reward, done, info = env.step(action)
            r_s += [reward]
            if done:
                break
            o_s += [observation]
        discounted_rewards = discount_rewards(r_s)
        normalized_rewards = (discounted_rewards-np.mean(discounted_rewards))/(np.std(discounted_rewards))
        sess.run(training_step, feed_dict={x:o_s, y_:a_s, rewards:normalized_rewards})
