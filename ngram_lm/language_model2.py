from nltk.lm import MLE
# from nltk import ngrams
# from nltk.probability import LidstoneProbDist, WittenBellProbDist
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
import pickle

TOKENIZED_CORPUS_DIR = '../bin/tokenized_corpus.pkl'
N_GRAM = 3

class NgramLanguageModel:
	def __init__(self, tokenized_corpus_dir, n_gram=N_GRAM):
		with open(tokenized_corpus_dir, 'rb') as pickle_in:
			tokenized_corpus = pickle.load(pickle_in, encoding='utf8')

		train_data, padded_sents = padded_everygram_pipeline(N_GRAM, tokenized_corpus)

		# Maximum Likelihood Estimator (MLE) model - Class for providing MLE ngram model scores.
		lm = MLE(n_gram)
		lm.fit(train_data, padded_sents)

		# print(lm.counts[['engineering']]['manager']) # P('manager'|'engineering')

		# print(lm.score('manager', 'engineering'.split()))  

		print(lm.unmasked_score('manager')) # P('manager'|'engineering')


def main():
	lm = NgramLanguageModel(TOKENIZED_CORPUS_DIR)

if __name__ == "__main__":
   main()


