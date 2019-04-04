from nltk.tokenize import ToktokTokenizer
from nltk.util import everygrams
from nltk import FreqDist
from nltk.lm import MLE
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pickle

TOKENIZED_CORPUS_DIR = '../bin/tokenized_corpus.pkl'
N_GRAM = 3

class NGRAM_LANGUAGE_MODEL:
	def __init__(self, tokenized_corpus_dir):
		with open(tokenized_corpus_dir, 'rb') as pickle_in:
			tokenized_corpus = pickle.load(pickle_in, encoding='utf8')

		# fd1 = FreqDist(corpus)

		#The n order of n-grams, if it's 2-grams, you pad once, 3-grams pad twice, etc.
		#To create this vocabulary we need to pad our sentences (just like for counting ngrams) and then combine the sentences into one flat stream of words.
		# print(list(flatten(pad_both_ends(sentence.split(), n=2) for sentence in corpus)))
		train_data, padded_sents = padded_everygram_pipeline(N_GRAM, tokenized_corpus)

		model = MLE(N_GRAM)
		model.fit(train_data, padded_sents)
		
		# print(model.vocab.lookup(tokenized_corpus[0]))
		# If we lookup the vocab on unseen sentences not from the training data, 
		# it automatically replace words not in the vocabulary with `<UNK>`.
		print(model.vocab.lookup('engineer java pflege lah .'.split()))

		# When it comes to ngram models the training boils down to counting up the ngrams from the training corpus.
		# print(model.counts)
		
		print(model.counts['manager'])

		print(model.counts[['engineering']]['manager'])
		print(model.score('manager', 'engineering'.split()))  # P('manager'|'engineering')

		print(model['manager'])
		

		# detokenize = TreebankWordDetokenizer().detokenize

		# def generate_sent(model, num_words, text_seed):
		# 	content = []
		# 	for token in model.generate(num_words, text_seed=text_seed):
		# 		if token == '<s>':
		# 			continue
		# 		if token == '</s>':
		# 			break
		# 		content.append(token)
		# 	return detokenize(content)

		# print(model.generate(2, text_seed=['engineer']))

		# print(generate_sent(model, 2, 'key'))

		# def predict(iterations, word):
		# 	for _ in range(iterations):
		# 		max_value = 0
		# 		for k, v in model[word].iteritems():
		# 			if v >= max_value:
		# 				word = k
		# 				max_value = v
		# 		sentence.append(word)
		# 	print(" ".join(sentence))
		# predict(2, 'manager')

def main():
	ngram_model = NGRAM_LANGUAGE_MODEL(TOKENIZED_CORPUS_DIR)

if __name__ == "__main__":
   main()