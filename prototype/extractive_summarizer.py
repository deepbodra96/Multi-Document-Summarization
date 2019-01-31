import nltk
import re
import heapq
import os
import math
import pprint

stopwords = nltk.corpus.stopwords.words('english')
pp = pprint.PrettyPrinter(indent=4)

total_sentences = 0
total_summary_sentences = 0

def get_files_in_dir(dir_name):
	return os.listdir(dir_name)


def to_continuous_string(text):
	text = re.sub(r'\[[0-9]*\]', ' ', text)  
	text = re.sub(r'\s+', ' ', text)
	return text


def remove_full_stop(text):
	text = re.sub('[^a-zA-Z]', ' ', text)  
	text = re.sub(r'\s+', ' ', text)  
	return text


def get_sentence_list(text):
	return nltk.sent_tokenize(text)


def gen_word_frequency_dict(text):
	word_frequencies = {}  
	for word in nltk.word_tokenize(text):  
		if word not in stopwords:
			if word not in word_frequencies.keys():
				word_frequencies[word] = 1
			else:
				word_frequencies[word] += 1
	return word_frequencies


def normalize_word_frequency_list(word_frequencies):
	maximum_frequncy = max(word_frequencies.values())
	for word in word_frequencies.keys():  
		word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)
	return word_frequencies


def calc_sentence_scores(sentence_list, word_frequencies):
	sentence_scores = {}  
	for sent in sentence_list:  
		for word in nltk.word_tokenize(sent.lower()):
			if word in word_frequencies.keys():
				if len(sent.split(' ')) < 30:
					if sent not in sentence_scores.keys():
						sentence_scores[sent] = word_frequencies[word]
					else:
						sentence_scores[sent] += word_frequencies[word]
	return sentence_scores


def get_top_n_sentences(n, sentence_scores):
	global total_summary_sentences

	summary_sentences = heapq.nlargest(n, sentence_scores, key=sentence_scores.get)
	total_summary_sentences += len(summary_sentences)
	summary = ' '.join(summary_sentences)
	return summary


def get_extractive_summary(text, summarization_ratio):
	global total_sentences

	article_text = to_continuous_string(text)
	formatted_article_text = remove_full_stop(article_text)

	sentence_list = get_sentence_list(article_text)
	total_sentences += len(sentence_list)

	word_frequencies = gen_word_frequency_dict(formatted_article_text)
	normalized_word_frequencies = normalize_word_frequency_list(word_frequencies)
	sentence_scores = calc_sentence_scores(sentence_list, normalized_word_frequencies)
	
	n_sentences = len(sentence_list)
	n_summary_sentences = math.ceil(n_sentences * summarization_ratio)

	summary = get_top_n_sentences(n_summary_sentences, sentence_scores)
	print("sentence_scores")
	pp.pprint(sentence_scores)
	print("\n-----\n")
	print("summary\n", summary)
	return summary


def main():

	test_cases_root_dir = 'test_cases'
	test_dir = input("Document Directory Name:")
	summarization_ratio = float(input("Summarization Ratio:"))

	# Concatenating Summaries
	# summary = ''
	# files_list = get_files_in_dir(test_cases_root_dir + '/' + test_dir)
	# for file in files_list:
	# 	article_text = ''
	# 	with open(test_cases_root_dir + '/' + test_dir + '/' + file, 'r') as myfile:
	# 		article_text = myfile.read().replace('\n', '')
	# 	summary += get_extractive_summary(article_text, summarization_ratio) + "\n\n"
	# with open(test_cases_root_dir + '/' + test_dir + '/' + 'summary.txt', 'w') as sum_file:
	# 	sum_file.write(summary)
	# print("Total Sentences = ", total_sentences)
	# print("Total Sentences after summary = ", total_summary_sentences)

	# Concatenating Articles
	files_list = get_files_in_dir(test_cases_root_dir + '/' + test_dir)
	article_text = ''
	for file in files_list:
		with open(test_cases_root_dir + '/' + test_dir + '/' + file, 'r') as myfile:
			article_text += myfile.read().replace('\n', '')
	print(article_text)
	summary = get_extractive_summary(article_text, summarization_ratio) + "\n\n"
	with open(test_cases_root_dir + '/' + test_dir + '/' + 'summary.txt', 'w') as sum_file:
		sum_file.write(summary)
	print("Total Sentences = ", total_sentences)
	print("Total Sentences after summary = ", total_summary_sentences)
	return

main()
