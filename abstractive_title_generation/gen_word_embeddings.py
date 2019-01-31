from collections import Counter
from itertools import chain

from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

import _pickle as pickle

import json

n_articles = 2500
with open('./data/sample-1M.jsonl') as fp:
	articles = [next(fp) for x in range(n_articles)]

heads, desc = [], []
for i in range(n_articles):
	article = json.loads(articles[i])
	idx = article['id']
	head = article['title']
	des = article['content']
	des = '.'.join([ sent.strip() for sent in des.split(".")[:3]])
	heads.append(head)
	desc.append(des)

embedding_dim = 100
vocab_size = 40000
seed = 42

# Build Vocabulary
def get_vocab(lst):
	vocab_count = Counter(w for txt in lst for w in txt.split())
	vocab = list(map(lambda x: x[0], sorted(vocab_count.items(), key=lambda x: -x[1])))
	return vocab, vocab_count

# vocab is a list of all the unique words
# vocab_count is a (Counter object) dictionary of all the words with its count
vocab, vocab_count = get_vocab(heads+desc)



# Index Words
empty = 0 # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos+1 # first real word

def get_idx(vocab):
	word2idx = dict((word, idx+start_idx) for idx, word in enumerate(vocab))
	word2idx['<empty>'] = empty
	word2idx['<eos>'] = eos
	
	idx2word = dict((idx, word) for word, idx in word2idx.items())

	return word2idx, idx2word

word2idx, idx2word = get_idx(vocab)


# Word Embedding using Glove

# read glove
glove_path = './glove.6B/glove.6B.%dd.txt' % (embedding_dim)
from subprocess import check_output
glove_n_symbols = check_output('find /c /v "%s" %s'%('', glove_path), shell=True).decode()
glove_n_symbols = int(glove_n_symbols.split()[-1])



glove_index_dict = {}
glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
globale_scale=0.1
with open(glove_path, 'r', encoding='utf-8') as fp:
	i = 0
	for l in fp:
		l = l.strip().split()
		w = l[0]
		glove_index_dict[w] = i
		glove_embedding_weights[i,:] = list(map(float, l[1:]))
		i += 1
glove_embedding_weights *= globale_scale


for w, i in glove_index_dict.items():
	w = w.lower()
	if w not in glove_index_dict:
		glove_index_dict[w] = i


# Embedding Matrix
np.random.seed(seed)
shape = (vocab_size, embedding_dim)
scale = glove_embedding_weights.std()*np.sqrt(12)/2 # uniform and not normal
embedding = np.random.uniform(low=-scale, high=scale, size=shape)

# copy from glove weights of words that appear in our short vocabulary (idx2word)
c = 0
for i in range(len(idx2word)):
	w = idx2word[i]
	g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
	if g is None and w.startswith('#'): # glove has no hastags (I think...)
		w = w[1:]
		g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
	if g is not None:
		embedding[i,:] = glove_embedding_weights[g,:]
		c+=1
# print('number of tokens, in small vocab, found in glove and copied to embedding', c,c/float(vocab_size))

glove_thr = 0.5
word2glove = {}
for w in word2idx:
	if w in glove_index_dict:
		g = w
	elif w.lower() in glove_index_dict:
		g = w.lower()
	elif w.startswith('#') and w[1:] in glove_index_dict:
		g = w[1:]
	elif w.startswith('#') and w[1:].lower() in glove_index_dict:
		g = w[1:].lower()
	else:
		continue
	word2glove[w] = g

normed_embedding = embedding/np.array([np.sqrt(np.dot(gweight,gweight)) for gweight in embedding])[:,None]
nb_unknown_words = 100

glove_match = []
for w, idx in word2idx.items():
	if idx >= vocab_size-nb_unknown_words and w.isalpha() and w in word2glove:
		gidx = glove_index_dict[word2glove[w]]
		gweight = glove_embedding_weights[gidx,:].copy()
		# find row in embedding that has the highest cos score with gweight
		gweight /= np.sqrt(np.dot(gweight,gweight))
		score = np.dot(normed_embedding[:vocab_size-nb_unknown_words], gweight)
		while True:
			embedding_idx = score.argmax()
			s = score[embedding_idx]
			if s < glove_thr:
				break
			if idx2word[embedding_idx] in word2glove :
				glove_match.append((w, embedding_idx, s)) 
				break
			score[embedding_idx] = -1
glove_match.sort(key = lambda x: -x[2])
print('# of glove substitutes found', len(glove_match))

# for orig, sub, score in glove_match[-10:]:
#     print(score, orig,'=>', idx2word[sub])


glove_idx2idx = dict((word2idx[w], embedding_idx) for  w, embedding_idx, _ in glove_match)

# Data
Y = [[word2idx[token] for token in headline.split()] for headline in heads]
X = [[word2idx[token] for token in d.split()] for d in desc]

# print('Y', len(Y))
# print('X', len(X))

# Plots
fig, ax = plt.subplots(2, 2)

# Plot 1
ax[0, 0].plot([vocab_count[w] for w in vocab]);
plt.gca().set_xscale("log", nonposx='clip')
plt.gca().set_yscale("log", nonposy='clip')

# plt.title('word distribution in headlines and discription')
# plt.xlabel('rank')
# plt.ylabel('total appearances')


# Plot 2
ax[0, 1].hist(list(map(len, Y)), bins=50)

# Plot 3
ax[1, 0].hist(list(map(len, X)), bins=50)

plt.show()

with open('%s.pkl'%('./vocabulary-embedding/vocabulary-embedding'),'wb') as fp:
	pickle.dump((embedding, idx2word, word2idx, glove_idx2idx),fp,-1)

with open('%s.data.pkl'%('./vocabulary-embedding/vocabulary-embedding'),'wb') as fp:
	pickle.dump((X,Y),fp,-1)