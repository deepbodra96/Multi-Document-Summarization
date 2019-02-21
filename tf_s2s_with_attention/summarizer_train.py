import tensorflow as tf
from tensorflow.contrib import rnn

from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec

from nltk.tokenize import word_tokenize

import numpy as np

import pickle

import os
import collections
import re
import time

start = time.perf_counter()

def clean_str(sentence):
	sentence = re.sub("[#.]+", "#", sentence)
	return sentence


def get_text_list(data_path, toy):
	with open (data_path, "r", encoding="utf-8") as f:
		if not toy:
			return [clean_str(x.strip()) for x in f.readlines()]
		else:
			return [clean_str(x.strip()) for x in f.readlines()][:500]


train_article_path = "sumdata/sent-comp/train01-desc.txt"
train_title_path = "sumdata/sent-comp/train01-head.txt"
valid_article_path = "sumdata/sent-comp/train01-head.txt"
valid_title_path = "sumdata/sent-comp/train01-head.txt"

config = {
			'embedding_size': 300,
			'num_hidden': 150,
			'num_layers': 2,
			'learning_rate': 1e-3,
			'beam_width':10,
			'keep_prob': 0.8,
			'glove': True,
			'batch_size': 64,
			'num_epochs': 10
}

def build_dict(step, toy=False):
	if step == "train":
		train_article_list = get_text_list(train_article_path, toy)
		train_title_list = get_text_list(train_title_path, toy)

		words = list()
		for sentence in train_article_list + train_title_list:
			for word in word_tokenize(sentence):
				words.append(word)

		word_counter = collections.Counter(words).most_common()
		word_dict = dict()
		word_dict["<padding>"] = 0
		word_dict["<unk>"] = 1
		word_dict["<s>"] = 2
		word_dict["</s>"] = 3
		for word, _ in word_counter:
			word_dict[word] = len(word_dict)

		with open("word_dict.pickle", "wb") as f:
			pickle.dump(word_dict, f)

	elif step == "valid":
		with open("word_dict.pickle", "rb") as f:
			word_dict = pickle.load(f)

	reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

	return word_dict, reversed_dict


def build_dataset(step, word_dict, article_max_len, summary_max_len, toy=False):
	if step == "train":
		article_list = get_text_list(train_article_path, toy)
		title_list = get_text_list(train_title_path, toy)
	elif step == "valid":
		article_list = get_text_list(valid_article_path, toy)
	else:
		raise NotImplementedError

	x = [word_tokenize(d) for d in article_list]
	x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in x]
	x = [d[:article_max_len] for d in x]
	x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]
	
	if step == "valid":
		return x
	else:        
		y = [word_tokenize(d) for d in title_list]
		y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in y]
		y = [d[:(summary_max_len - 1)] for d in y]
		return x, y


def get_init_embedding(reversed_dict, embedding_size):
	glove_file = "./glove.6B/glove.6B.300d.txt"
	word2vec_file = get_tmpfile("word2vec_format.vec")
	glove2word2vec(glove_file, word2vec_file)
	print("Loading Glove vectors...")
	word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

	word_vec_list = list()
	for _, word in sorted(reversed_dict.items()):
		try:
			word_vec = word_vectors.word_vec(word)
		except KeyError:
			word_vec = np.zeros([embedding_size], dtype=np.float32)

		word_vec_list.append(word_vec)

	# Assign random vector to <s>, </s> token
	word_vec_list[2] = np.random.normal(0, 1, embedding_size)
	word_vec_list[3] = np.random.normal(0, 1, embedding_size)

	return np.array(word_vec_list)


def batch_iter(inputs, outputs, batch_size, num_epochs):
	inputs = np.array(inputs)
	outputs = np.array(outputs)

	num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
	for epoch in range(num_epochs):
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, len(inputs))
			yield inputs[start_index:end_index], outputs[start_index:end_index]


class Model(object):
	def __init__(self, reversed_dict, article_max_len, summary_max_len, config, forward_only=False):
		self.vocabulary_size = len(reversed_dict)
		self.embedding_size = config['embedding_size']
		self.num_hidden = config['num_hidden']
		self.num_layers = config['num_layers']
		self.learning_rate = config['learning_rate']
		self.beam_width = config['beam_width']
		if not forward_only:
			self.keep_prob = config['keep_prob']
		else:
			self.keep_prob = 1.0
		self.cell = tf.nn.rnn_cell.BasicLSTMCell
		with tf.variable_scope("decoder/projection"):
			self.projection_layer = tf.layers.Dense(self.vocabulary_size, use_bias=False)

		self.batch_size = tf.placeholder(tf.int32, (), name="batch_size")
		self.X = tf.placeholder(tf.int32, [None, article_max_len])
		self.X_len = tf.placeholder(tf.int32, [None])
		self.decoder_input = tf.placeholder(tf.int32, [None, summary_max_len])
		self.decoder_len = tf.placeholder(tf.int32, [None])
		self.decoder_target = tf.placeholder(tf.int32, [None, summary_max_len])
		self.global_step = tf.Variable(0, trainable=False)

		with tf.name_scope("embedding"):
			if not forward_only and config['glove']:
				init_embeddings = tf.constant(get_init_embedding(reversed_dict, self.embedding_size), dtype=tf.float32)
			else:
				init_embeddings = tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0)
			self.embeddings = tf.get_variable("embeddings", initializer=init_embeddings)
			self.encoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.X), perm=[1, 0, 2])
			self.decoder_emb_inp = tf.transpose(tf.nn.embedding_lookup(self.embeddings, self.decoder_input), perm=[1, 0, 2])

		with tf.name_scope("encoder"):
			fw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
			bw_cells = [self.cell(self.num_hidden) for _ in range(self.num_layers)]
			fw_cells = [rnn.DropoutWrapper(cell) for cell in fw_cells]
			bw_cells = [rnn.DropoutWrapper(cell) for cell in bw_cells]

			encoder_outputs, encoder_state_fw, encoder_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
				fw_cells, bw_cells, self.encoder_emb_inp,
				sequence_length=self.X_len, time_major=True, dtype=tf.float32)
			self.encoder_output = tf.concat(encoder_outputs, 2)
			encoder_state_c = tf.concat((encoder_state_fw[0].c, encoder_state_bw[0].c), 1)
			encoder_state_h = tf.concat((encoder_state_fw[0].h, encoder_state_bw[0].h), 1)
			self.encoder_state = rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

		with tf.name_scope("decoder"), tf.variable_scope("decoder") as decoder_scope:
			decoder_cell = self.cell(self.num_hidden * 2)

			if not forward_only:
				attention_states = tf.transpose(self.encoder_output, [1, 0, 2])
				attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
					self.num_hidden * 2, attention_states, memory_sequence_length=self.X_len, normalize=True)
				decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
																   attention_layer_size=self.num_hidden * 2)
				initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
				initial_state = initial_state.clone(cell_state=self.encoder_state)
				helper = tf.contrib.seq2seq.TrainingHelper(self.decoder_emb_inp, self.decoder_len, time_major=True)
				decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
				outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, scope=decoder_scope)
				self.decoder_output = outputs.rnn_output
				self.logits = tf.transpose(
					self.projection_layer(self.decoder_output), perm=[1, 0, 2])
				self.logits_reshape = tf.concat(
					[self.logits, tf.zeros([self.batch_size, summary_max_len - tf.shape(self.logits)[1], self.vocabulary_size])], axis=1)
			else:
				tiled_encoder_output = tf.contrib.seq2seq.tile_batch(
					tf.transpose(self.encoder_output, perm=[1, 0, 2]), multiplier=self.beam_width)
				tiled_encoder_final_state = tf.contrib.seq2seq.tile_batch(self.encoder_state, multiplier=self.beam_width)
				tiled_seq_len = tf.contrib.seq2seq.tile_batch(self.X_len, multiplier=self.beam_width)
				attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
					self.num_hidden * 2, tiled_encoder_output, memory_sequence_length=tiled_seq_len, normalize=True)
				decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
																   attention_layer_size=self.num_hidden * 2)
				initial_state = decoder_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size * self.beam_width)
				initial_state = initial_state.clone(cell_state=tiled_encoder_final_state)
				decoder = tf.contrib.seq2seq.BeamSearchDecoder(
					cell=decoder_cell,
					embedding=self.embeddings,
					start_tokens=tf.fill([self.batch_size], tf.constant(2)),
					end_token=tf.constant(3),
					initial_state=initial_state,
					beam_width=self.beam_width,
					output_layer=self.projection_layer
				)
				outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
					decoder, output_time_major=True, maximum_iterations=summary_max_len, scope=decoder_scope)
				self.prediction = tf.transpose(outputs.predicted_ids, perm=[1, 2, 0])

		with tf.name_scope("loss"):
			if not forward_only:
				crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
					logits=self.logits_reshape, labels=self.decoder_target)
				weights = tf.sequence_mask(self.decoder_len, summary_max_len, dtype=tf.float32)
				self.loss = tf.reduce_sum(crossent * weights / tf.to_float(self.batch_size))

				params = tf.trainable_variables()
				gradients = tf.gradients(self.loss, params)
				clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
				optimizer = tf.train.AdamOptimizer(self.learning_rate)
				self.update = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)


article_max_len = 50
summary_max_len = 25
toy = True

# Main code
if not os.path.exists("saved_model"):
	os.mkdir("saved_model")
else:
	old_model_checkpoint_path = open('saved_model/checkpoint', 'r')
	old_model_checkpoint_path = "".join(["saved_model/",old_model_checkpoint_path.read().splitlines()[0].split('"')[1] ])


print("Building dictionary...")
word_dict, reversed_dict = build_dict("train", toy)
print("Loading training dataset...")
train_x, train_y = build_dataset("train", word_dict, article_max_len, summary_max_len, toy)

with tf.Session() as sess:
	model = Model(reversed_dict, article_max_len, summary_max_len, config)
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver(tf.global_variables())
	if 'old_model_checkpoint_path' in globals():
		print("Continuing from previous trained model:" , old_model_checkpoint_path , "...")
		saver.restore(sess, old_model_checkpoint_path )

	batches = batch_iter(train_x, train_y, config['batch_size'], config['num_epochs'])
	num_batches_per_epoch = (len(train_x) - 1) // config['batch_size'] + 1

	print("\nIteration starts.")
	print("Number of batches per epoch :", num_batches_per_epoch)
	for batch_x, batch_y in batches:
		batch_x_len = list(map(lambda x: len([y for y in x if y != 0]), batch_x))
		batch_decoder_input = list(map(lambda x: [word_dict["<s>"]] + list(x), batch_y))
		batch_decoder_len = list(map(lambda x: len([y for y in x if y != 0]), batch_decoder_input))
		batch_decoder_output = list(map(lambda x: list(x) + [word_dict["</s>"]], batch_y))

		batch_decoder_input = list(
			map(lambda d: d + (summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_input))
		batch_decoder_output = list(
			map(lambda d: d + (summary_max_len - len(d)) * [word_dict["<padding>"]], batch_decoder_output))

		train_feed_dict = {
			model.batch_size: len(batch_x),
			model.X: batch_x,
			model.X_len: batch_x_len,
			model.decoder_input: batch_decoder_input,
			model.decoder_len: batch_decoder_len,
			model.decoder_target: batch_decoder_output
		}

		_, step, loss = sess.run([model.update, model.global_step, model.loss], feed_dict=train_feed_dict)

		if step % 1000 == 0:
			print("step {0}: loss = {1}".format(step, loss))

		if step % num_batches_per_epoch == 0:
			hours, rem = divmod(time.perf_counter() - start, 3600)
			minutes, seconds = divmod(rem, 60)
			saver.save(sess, "./saved_model/model.ckpt", global_step=step)
			print(" Epoch {0}: Model is saved.".format(step // num_batches_per_epoch),
			"Elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds) , "\n")
