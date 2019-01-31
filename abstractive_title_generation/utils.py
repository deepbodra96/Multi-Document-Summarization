from tensorflow.python.keras.preprocessing import sequence

from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras import backend as K

from IPython.core.display import display, HTML

import numpy as np

import Levenshtein

def prt(label, x):
	print(label+':', end=' ')
	for w in x:
		print(idx2word[w], end=' ')
	print()


def str_shape(x):
	return 'x'.join(map(str,x.shape))


def inspect_model(model):
	for i,l in enumerate(model.layers):
		print(i, 'cls=%s name=%s'%(type(l).__name__, l.name))
		weights = l.get_weights()
		for weight in weights:
			print(str_shape(weight), end=' ')

maxlend=25
maxlenh=25
maxlen = maxlend + maxlenh
activation_rnn_size = 40 if maxlend else 0
empty=0
eos = 1

batch_size=64

def simple_context(X, mask,  n=activation_rnn_size, maxlend=maxlend, maxlenh=maxlenh):
	desc, head = X[:,:maxlend,:], X[:,maxlend:,:]
	head_activations, head_words = head[:,:,:n], head[:,:,n:]
	desc_activations, desc_words = desc[:,:,:n], desc[:,:,n:]
	
	# RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
	# activation for every head word and every desc word
	activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2,2))
	# make sure we dont use description words that are masked out
	activation_energies = activation_energies + -1e20*K.expand_dims(1.-K.cast(mask[:, :maxlend],'float32'),1)
	
	# for every head word compute weights for every desc word
	activation_energies = K.reshape(activation_energies,(-1,maxlend))
	activation_weights = K.softmax(activation_energies)
	activation_weights = K.reshape(activation_weights,(-1,maxlenh,maxlend))

	# for every head word compute weighted average of desc words
	desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2,1))
	return K.concatenate((desc_avg_word, head_words))



class SimpleContext(Lambda):
	def __init__(self,**kwargs):
		super(SimpleContext, self).__init__(simple_context,**kwargs)
		self.supports_masking = True

	def compute_mask(self, input, input_mask=None):
		return input_mask[:, maxlend:]
	
	def get_output_shape_for(self, input_shape):
		nb_samples = input_shape[0]
		n = 2*(rnn_size - activation_rnn_size)
		return (nb_samples, maxlenh, n)


def lpadd(x, maxlend=maxlend, eos=eos):
	"""left (pre) pad a description to maxlend and then add eos.
	The eos is the input to predicting the first word in the headline
	"""
	assert maxlend >= 0
	if maxlend == 0:
		return [eos]
	n = len(x)
	if n > maxlend:
		x = x[-maxlend:]
		n = maxlend
	return [empty]*(maxlend-n) + x + [eos]


# Sample Generation
# variation to https://github.com/ryankiros/skip-thoughts/blob/master/decoding/search.py

def beamsearch(model, vocab_size, nb_unknown_words, predict, start=[empty]*maxlend + [eos], avoid=None, avoid_score=1,
			   k=1, maxsample=maxlen, use_unk=True, oov=None, empty=empty, eos=eos, temperature=1.0):
	"""return k samples (beams) and their NLL scores, each sample is a sequence of labels,
	all samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
	You need to supply `predict` which returns the label probability of each sample.
	`use_unk` allow usage of `oov` (out-of-vocabulary) label in samples
	"""
	if oov is None:
		oov = vocab_size-1 
	def sample(energy, n, temperature=temperature):
		"""sample at most n different elements according to their energy"""
		n = min(n,len(energy))
		prb = np.exp(-np.array(energy) / temperature )
		res = []
		for i in range(n):
			z = np.sum(prb)
			r = np.argmax(np.random.multinomial(1, prb/z, 1))
			res.append(r)
			prb[r] = 0. # make sure we select each element only once
		return res

	dead_samples = []
	dead_scores = []
	live_samples = [list(start)]
	live_scores = np.array([0])

	while live_samples:
		# for every possible live sample calc prob for every possible label 
		probs = predict(live_samples, model, empty=empty)
		# print('live_samples', live_samples)
		# assert vocab_size == probs.shape[1]

		# total score for every sample is sum of -log of word prb
		# print('probs', probs)
		cand_scores = np.array(live_scores)[:,None] - np.log(np.absolute(probs))
		cand_scores[:,empty] = 1e20
		if not use_unk and oov is not None:
			cand_scores[:,oov] = 1e20
		if avoid:
			for a in avoid:
				for i, s in enumerate(live_samples):
					n = len(s) - len(start)
					if n < len(a):
						# at this point live_sample is before the new word,
						# which should be avoided, is added
						cand_scores[i,a[n]] += avoid_score
		live_scores = list(cand_scores.flatten())
		

		# find the best (lowest) scores we have from all possible dead samples and
		# all live samples and all possible new words added
		scores = dead_scores + live_scores
		ranks = sample(scores, k)
		n = len(dead_scores)
		dead_scores = [dead_scores[r] for r in ranks if r < n]
		dead_samples = [dead_samples[r] for r in ranks if r < n]
		
		live_scores = [live_scores[r-n] for r in ranks if r >= n]
		live_samples = [live_samples[(r-n)//vocab_size]+[(r-n)%vocab_size] for r in ranks if r >= n]

		# live samples that should be dead are...
		# even if len(live_samples) == maxsample we dont want it dead because we want one
		# last prediction out of it to reach a headline of maxlenh
		def is_zombie(s):
			return s[-1] == eos or len(s) > maxsample
		
		# add zombies to the dead
		dead_scores += [c for s, c in zip(live_samples, live_scores) if is_zombie(s)]
		dead_samples += [s for s in live_samples if is_zombie(s)]
		
		# remove zombies from the living 
		live_scores = [c for s, c in zip(live_samples, live_scores) if not is_zombie(s)]
		live_samples = [s for s in live_samples if not is_zombie(s)]

	return dead_samples, dead_scores


def keras_rnn_predict(samples, model, empty=empty, maxlen=maxlen):
	"""for every sample, calculate probability for every possible label
	you need to supply your RNN model and maxlen - the length of sequences it can handle
	"""
	sample_lengths = list(map(len, samples))
	assert all(l > maxlend for l in sample_lengths)
	assert all(l[maxlend] == eos for l in samples)
	# pad from right (post) so the first maxlend will be description followed by headline
	data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
	probs = model.predict(data, verbose=0, batch_size=batch_size)
	t =  np.array([prob[sample_length-maxlend-1] for prob, sample_length in zip(probs, sample_lengths)])
	return t


def vocab_fold(xs, vocab_size, nb_unknown_words):
	"""convert list of word indexes that may contain words outside vocab_size to words inside.
	If a word is outside, try first to use glove_idx2idx to find a similar word inside.
	If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
	"""
	xs = [x if x < vocab_size-nb_unknown_words else glove_idx2idx.get(x,x) for x in xs]
	# the more popular word is <0> and so on
	outside = sorted([x for x in xs if x >= vocab_size-nb_unknown_words])
	# if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
	outside = dict((x,vocab_size-1-min(i, nb_unknown_words-1)) for i, x in enumerate(outside))
	xs = [outside.get(x,x) for x in xs]
	return xs


def vocab_unfold(desc,xs, vocab_size, nb_unknown_words):
	# assume desc is the unfolded version of the start of xs
	unfold = {}
	for i, unfold_idx in enumerate(desc):
		fold_idx = xs[i]
		if fold_idx >= vocab_size-nb_unknown_words:
			unfold[fold_idx] = unfold_idx
	return [unfold.get(x,x) for x in xs]


def gensamples(model, word2idx, idx2word, glove_idx2idx, vocab_size, nb_unknown_words, X=None, X_test=None, Y_test=None, avoid=None, avoid_score=1, skips=2, k=10, batch_size=batch_size, short=True, temperature=1., use_unk=True):
	if X is None or isinstance(X,int):
		if X is None:
			i = random.randint(0,len(X_test)-1)
		else:
			i = X
		print('HEAD %d:'%i,' '.join(idx2word[w] for w in Y_test[i]), end=' ')
		print('DESC:',' '.join(idx2word[w] for w in X_test[i]), end=' ')
		sys.stdout.flush()
		x = X_test[i]
	else:
		x = [word2idx[w.rstrip('^')] for w in X.split()]
		
	if avoid:
		# avoid is a list of avoids. Each avoid is a string or list of word indeicies
		if isinstance(avoid,str) or isinstance(avoid[0], int):
			avoid = [avoid]
		avoid = [a.split() if isinstance(a,str) else a for a in avoid]
		avoid = [vocab_fold([w if isinstance(w,int) else word2idx[w] for w in a], vocab_size, nb_unknown_words)
				 for a in avoid]

	print('HEADS:', end=' ')
	samples = []
	if maxlend == 0:
		skips = [0]
	else:
		skips = range(min(maxlend,len(x)), max(maxlend,len(x)), abs(maxlend - len(x)) // skips + 1)
	for s in skips:
		start = lpadd(x[:s])
		fold_start = vocab_fold(start, vocab_size, nb_unknown_words)
		sample, score = beamsearch(model, vocab_size, nb_unknown_words, predict=keras_rnn_predict, start=fold_start, avoid=avoid, avoid_score=avoid_score,
								   k=k, temperature=temperature, use_unk=use_unk)
		assert all(s[maxlend] == eos for s in sample)
		samples += [(s,start,scr) for s,scr in zip(sample,score)]

	samples.sort(key=lambda x: x[-1])
	codes = []
	for sample, start, score in samples:
		code = ''
		words = []
		sample = vocab_unfold(start, sample, vocab_size, nb_unknown_words)[len(start):]
		for w in sample:
			if w == eos:
				break
			words.append(idx2word[w])
			code += chr(w//(256*256)) + chr((w//256)%256) + chr(w%256)
		if short:
			distance = min([100] + [-Levenshtein.jaro(code,c) for c in codes])
			if distance > -0.6:
				print(score, ' '.join(words))
		#         print '%s (%.2f) %f'%(' '.join(words), score, distance)
		else:
				print(score, ' '.join(words))
		codes.append(code)
	return samples


def heat(sample,weights,dark=0.3):
    weights = (weights - weights.min())/(weights.max() - weights.min() + 1e-4)
    html = ''
    fmt = ' <span style="background-color: #{0:x}{0:x}ff">{1}</span>'
    for t,w in zip(sample,weights):
        c = int(256*((1.-dark)*(1.-w)+dark))
        html += fmt.format(c,idx2word[t])
    display(HTML(html))