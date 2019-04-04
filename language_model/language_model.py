from nltk.lm import MLE
# from nltk.probability import LidstoneProbDist, WittenBellProbDist, ConditionalFreqDist
from nltk.lm.preprocessing import pad_both_ends, flatten, padded_everygram_pipeline
import pickle
from collections import defaultdict

TOKENIZED_CORPUS_DIR = '../bin/tokenized_corpus.pkl'
NGRAMS_DIR = '../bin/ngrams.pkl'
N_GRAM = 3

class NgramLanguageModel:
	@staticmethod
	def calc_likelihood(tokenized_corpus_dir=TOKENIZED_CORPUS_DIR, ngrams_dir=NGRAMS_DIR, n_gram=N_GRAM):
		with open(tokenized_corpus_dir, 'rb') as pickle_in:
			tokenized_corpus = pickle.load(pickle_in, encoding='utf8')

		with open(ngrams_dir, 'rb') as pickle_in:
			ngrams = pickle.load(pickle_in, encoding='utf8')

		train_data, padded_sents = padded_everygram_pipeline(N_GRAM, tokenized_corpus)
		lm = MLE(n_gram) # Maximum Likelihood Estimator (MLE) model
		lm.fit(train_data, padded_sents)	
		likelihoods = defaultdict(list)

		# Likelihood estimator for generic ngrams (excluding unigrams)
		for k in ngrams:
			if k != 1:
				for ng in ngrams[k]:
					# gram is deteremined by  the number of slpits it has in its sentence
					tokens = ng.split(' ')
					# Score a word given some optional context. Unseen words are assigned probability 0.
					x, y = tokens[-1], tuple(tokens[:-1])
					score = lm.unmasked_score(x, context=y) # P('x'|'y')
					if score > 0: #Smoothing can potentially be applied to improve results
						# we create a mapping of given word y to a list of possible next words and their scores
						likelihoods[y].append((score, x)) 

		with open('../bin/likelihoods.pkl', 'wb') as output:
			pickle.dump(likelihoods, output)
			output.close()

def main():
	NgramLanguageModel.calc_likelihood()

if __name__ == "__main__":
   main()


