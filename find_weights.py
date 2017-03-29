import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist

from word2vec_pipeline.utils.data_utils import load_w2vec
from word2vec_pipeline.utils.db_utils import item_iterator
from word2vec_pipeline import simple_config
from tqdm import tqdm

config = simple_config.load()
target_column = config['target_column']

save_dest = config["score"]["output_data_directory"]
f_db = config["score"]["score_word_document_dispersion"]["f_db"]
f_save = os.path.join(save_dest, f_db)



W = load_w2vec()
n_vocab,n_dim = W.syn0.shape
word2index = dict(zip(W.index2word, range(n_vocab)))

batch_size = 2**8
n_epochs = 100

##################################################################

# Tensorflow model here

import tensorflow as tf

print "Building model"

X = tf.placeholder(tf.float32, shape=[batch_size, n_vocab])
word2vec_layer = tf.constant(W.syn0,shape=(n_vocab,n_dim))

e1 = tf.exp(1.0)
alpha = tf.Variable(tf.ones([n_vocab])/e1)
#alpha = tf.Variable(tf.ones([n_vocab]))

alpha = tf.clip_by_value(alpha, -2, 2.0)
Y = X * tf.exp(alpha) / e1
#Y = X * alpha

Y = tf.matmul(Y, word2vec_layer)
Y = tf.nn.l2_normalize(Y, dim=1)
dist = tf.matmul(Y, tf.transpose(Y))


#dist = tf.clip_by_value(dist,0,1)
#loss = (tf.reduce_sum(dist) - batch_size) / (batch_size**2-batch_size)

# Choose only the largest distance to minimize
#dist = tf.reduce_max(dist, axis=0)
#loss = tf.reduce_sum(dist) / batch_size

# Push out the loss in the middle
#dist = tf.matrix_set_diag(dist, [0,]*batch_size)
#dist = dist*tf.cos(dist*pi/2)
#loss = tf.reduce_sum(dist) / (batch_size**2-batch_size)

# Push out the loss in the middle
dist = tf.matrix_set_diag(dist, [0,]*batch_size)
dist = tf.cos(dist*pi/2)
loss = tf.reduce_sum(dist) / (batch_size**2-batch_size)

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

print "Model building complete"

##################################################################


score_config = simple_config.load()["score"]
f_csv = os.path.join(
    score_config["output_data_directory"],
    score_config["term_document_frequency"]["f_db"],
)
IDF = pd.read_csv(f_csv)
IDF = dict(zip(IDF["word"].values, IDF["count"].values))
corpus_N = IDF.pop("__pipeline_document_counter")

# Compute the IDF
for key in IDF:
    IDF[key] = np.log(float(corpus_N) / (IDF[key] + 1))
IDF = [IDF[w] if w in IDF else 0.0 for w in W.index2word]
IDF = np.array(IDF)


V = []
for item in item_iterator():
    tokens = item[target_column].split()
    row = np.zeros(n_vocab)
    for w in tokens:
        if w not in word2index:
            continue
        row[word2index[w]] += 1
    V.append(row)

V = np.array(V)

V[V>1] = 1.0
V = V*IDF

n_samples = V.shape[0]

import random

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    random.shuffle(V)

    for k in xrange(n_epochs):

        epoch_loss = []

        for i in range(n_samples//batch_size - 1):
            
            v_batch = V[i*batch_size:(i+1)*batch_size]
            funcs = [optimizer, loss]
            _, ls = sess.run(funcs, feed_dict={X:v_batch})
            epoch_loss.append(ls)
            

        funcs = [alpha,]
        a, = sess.run(funcs, feed_dict={X:v_batch})

        print k,np.mean(epoch_loss), a.max(), a.min()
        
        if k and k%50==0:
            df = pd.DataFrame()
            df["words"] = W.index2word
            df["alpha"] = a
            df.set_index("words").to_csv(f_save)
            print df



            
#####################################################################
# numpy CPU VERSION FOR REFERENCE
#####################################################################

'''
batch_size = len(V)

def objective_func(alpha):
    X = normalize((V*alpha).dot(W.syn0))
    dist = pdist(X, metric='cosine')
    loss = dist.sum()
    loss /= (batch_size*(batch_size-1))/2.0
    print loss,alpha.max(), alpha.min()

    return loss

from scipy.optimize import minimize, fmin

alpha = np.ones(n_vocab)

# Starts off at 31.19, 11.28
print minimize(objective_func, alpha)#, method='Nelder-Mead')
'''
