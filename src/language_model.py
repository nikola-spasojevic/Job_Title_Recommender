from nltk.lm import MLE, Laplace
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
from collections import defaultdict
from .helper.tokenizer import Tokenizer
from .text_preprocessing import TEST_CORPUS_DIR, TRAIN_CORPUS_DIR
from .bag_of_words import UNIGRAM_FREQ_DIR, NGRAMS_DIR
import pickle

N_GRAM = 3

class NgramLanguageModel:
	@staticmethod
	def likelihoods_gen(ngrams_dir=NGRAMS_DIR, n_gram=N_GRAM):
		with open(ngrams_dir, 'rb') as pickle_in:
			ngrams = pickle.load(pickle_in, encoding='utf8')

		tokenized_train_corpus = Tokenizer.job_data_tokenizer(TRAIN_CORPUS_DIR)
		train_data, padded_sents = padded_everygram_pipeline(N_GRAM, tokenized_train_corpus)
		# Maximum Likelihood Estimator (MLE) model using Laplace Smoothing (gamma is always 1).
		lm = Laplace(n_gram)
		lm.fit(train_data, padded_sents)	
		likelihoods = defaultdict(list)

		# Likelihood estimator for ngrams
		for k in ngrams:
			for ng in ngrams[k]:
				# ngram is deteremined by  the number of splits it has in its sentence
				tokens = ng.split(' ')
				# Score a word given some optional context. Unseen words are assigned probability 0.
				x, y = tokens[-1], tuple(tokens[:-1])
				score = lm.unmasked_score(x, context=y) # P('x'|'y')
				# we create a mapping of given word y to a list of possible next words and their scores
				if score != 0:
					likelihoods[y].append((score, x)) 

		with open('bin/likelihoods.pkl', 'wb') as output:
			pickle.dump(likelihoods, output)
			output.close()

		def evaluate():
			with open(TEST_CORPUS_DIR, 'rb') as pickle_in:
				test_corpus = pickle.load(pickle_in, encoding='utf8')

		# Evaluate the total entropy of a corpus with respect to the model.
		# This is the sum of the log probability of each word in the test corpus.
			file = open('bin/model_evaluation.txt', 'w')
			file.write('Model Evaluation Score (Entropy): {}'.format(lm.entropy(test_corpus)))
			file.close()

		evaluate()

def main():
	NgramLanguageModel.likelihoods_gen(ngrams_dir='../'+NGRAMS_DIR)

if __name__ == "__main__":
   main()


