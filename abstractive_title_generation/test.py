from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.utils import to_categorical

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Embedding, Dense, Activation, Dropout, RepeatVector, Lambda, TimeDistributed
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.optimizers import Adam, RMSprop
from tensorflow.python.keras import backend as K

import pandas as pd
import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import _pickle as pickle
import h5py

import random, sys


from utils import prt, str_shape, inspect_model, simple_context, SimpleContext, lpadd, beamsearch, keras_rnn_predict, vocab_fold, vocab_unfold, gensamples, heat

maxlend=25 # 0 - if we dont want to use description at all
maxlenh=25
maxlen = maxlend + maxlenh
rnn_size = 128 # must be same as 160330-word-gen
rnn_layers = 1  # match FN1
batch_norm=False

activation_rnn_size = 40 if maxlend else 0

# training parameters
seed=42
p_W, p_U, p_dense, p_emb, weight_decay = 0.2, 0.2, 0.2, 0.2, 0.2
optimizer = 'rmsprop'
batch_size=64

nb_train_samples = 2
nb_val_samples = 1

# Read Word Embedding
with open('vocabulary-embedding/vocabulary-embedding.pkl', 'rb') as fp:
	embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
vocab_size, embedding_size = embedding.shape

nb_unknown_words = 10

for i in range(nb_unknown_words):
	idx2word[vocab_size-1-i] = '<%d>'%i

for i in range(vocab_size-nb_unknown_words, len(idx2word)):
	idx2word[i] = idx2word[i]+'^'

empty = 0
eos = 1
idx2word[empty] = '_'
idx2word[eos] = '~'

random.seed(seed)
np.random.seed(seed)

regularizer = l2(weight_decay) if weight_decay else None

model = Sequential()
model.add(Embedding(vocab_size, embedding_size,
					input_length=maxlen,
					embeddings_regularizer=regularizer, weights=[embedding], mask_zero=True,
					name='embedding_1'))
for i in range(rnn_layers):
	lstm = LSTM(rnn_size, return_sequences=True, # batch_norm=batch_norm,
				kernel_regularizer=regularizer, recurrent_regularizer=regularizer,
				bias_regularizer=regularizer, dropout=p_W, recurrent_dropout=p_U,
				name='lstm_%d'%(i+1))

	model.add(lstm)
	# model.add(Dropout(p_dense ,name='dropout_%d'%(i+1)))

# Load
def load_weights(model, filepath):
	"""Modified version of keras load_weights that loads as much as it can
	if there is a mismatch between file and model. It returns the weights
	of the first layer in which the mismatch has happened
	"""
	print('Loading', filepath, 'to', model.name)
	flattened_layers = model.layers
	with h5py.File(filepath, mode='r') as f:
		# new file format
		layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

		# we batch weight value assignments in a single backend call
		# which provides a speedup in TensorFlow.
		weight_value_tuples = []
		for name in layer_names:
			print(name)
			g = f[name]
			weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
			if len(weight_names):
				weight_values = [g[weight_name] for weight_name in weight_names]
				try:
					layer = model.get_layer(name=name)
				except:
					layer = None
				if not layer:
					print('failed to find layer', name, 'in model')
					print('weights', ' '.join(str_shape(w) for w in weight_values))
					print('stopping to load all other layers')
					weight_values = [np.array(w) for w in weight_values]
					break
				symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
				weight_value_tuples += zip(symbolic_weights, weight_values)
				weight_values = None
		K.batch_set_value(weight_value_tuples)
	return weight_values


weights = load_weights(model, './models/train.hdf5')
# print([w.shape for w in weights])

context_weight = K.variable(1.)
head_weight = K.variable(1.)
cross_weight = K.variable(0.)

if activation_rnn_size:
	model.add(SimpleContext(name='simplecontext_1'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

n = 2*(rnn_size - activation_rnn_size)


# out very own softmax
def output2probs(output):
	output = np.dot(output, weights[0]) + weights[1]
	output -= output.max()
	output = np.exp(output)
	output /= output.sum()
	return output


def output2probs1(output):
	output0 = np.dot(output[:n//2], weights[0][:n//2,:])
	output1 = np.dot(output[n//2:], weights[0][n//2:,:])
	output = output0 + output1 # + output0 * output1
	output += weights[1]
	output -= output.max()
	output = np.exp(output)
	output /= output.sum()
	return output

# samples = [lpadd([3]*26)]
# data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
# print(np.all(data[:,maxlend] == eos))
# print(data.shape, list(map(len, samples)))
# probs = model.predict(data, verbose=0, batch_size=1)
# print(probs.shape, vocab_size)


seed = 8
random.seed(seed)
np.random.seed(seed)


# X='we have not been to school nor have we College'
X = "VETERANS saluted Worcester's first ever breakfast club for ex-soldiers which won over hearts, minds and bellies.The Worcester Breakfast Club for HM Forces Veterans met at the Postal Order in Foregate Street at 10am on Saturday.The club is designed to allow veterans a place to meet, socialise, eat and drink, giving hunger and loneliness their marching orders"
# X= "VETERANS saluted Worcester's first ever breakfast club for ex-soldiers"
# samples = gensamples(model, word2idx, idx2word, glove_idx2idx, vocab_size, nb_unknown_words, X=X, skips=4, batch_size=batch_size, k=10, temperature=1.)
samples = gensamples(model, word2idx, idx2word, glove_idx2idx, vocab_size, nb_unknown_words, X=X, skips=2, batch_size=batch_size, k=10, temperature=1., use_unk=True, short=False)

sample = samples[0][0]

data = sequence.pad_sequences([sample], maxlen=maxlen, value=empty, padding='post', truncating='post')
print('data.shape', data.shape)

startd = np.where(data[0,:] != empty)[0][0]
lenh = np.where(data[0, maxlend+1:] == eos)[0][0]

# sent_len = np.where(data[0] == eos)[0][0]

print('startd, lenh', startd, lenh)

weights = model.predict(data, verbose=0, batch_size=1)
print('weights.shape', weights.shape)

plt.hist(np.array(weights[0,:lenh,startd:].flatten()+1), bins=100);

columns = [idx2word[data[0,i]] for i in range(startd, maxlend)]
rows = [idx2word[data[0,i]] for i in range(maxlend+1, maxlend+lenh+1)]

print('len(rows) len(columns)', len(rows), len(columns))
df = pd.DataFrame(weights[0, :lenh, :25], columns=columns, index=rows)

sns.heatmap(df);
plt.show()