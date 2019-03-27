from nltk.tag import pos_tag

def rectify_proper_nouns(ip_sents, op_sents):
	corrected_sents = []
	for ip_sent, op_sent in zip(ip_sents, op_sents):
		tagged_ip_sent = pos_tag(ip_sent.split())
		tagged_op_sent = pos_tag(op_sent.split())
		proper_nouns_ip_sent = [tagged_word[0] for tagged_word in tagged_ip_sent if tagged_word[1] == "NNP"]
		print('tagged_ip_sent, tagged_op_sent', tagged_ip_sent, tagged_op_sent)

		correct_proper_noun_idx = 0
		for i in range(len(tagged_op_sent)):
			predicted_word_tag = tagged_op_sent[i]
			if predicted_word_tag[1] == 'NNP':
				op_sent.replace(predicted_word_tag[0], proper_nouns_ip_sent[correct_proper_noun_idx], 1)
				correct_proper_noun_idx+=1
		corrected_sents.append(op_sent)
	return corrected_sents