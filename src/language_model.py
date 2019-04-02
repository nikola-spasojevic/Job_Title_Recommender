from nltk.corpus import reuters
from nltk import bigrams, trigrams
from collections import Counter, defaultdict
import pickle

class LM:

	def __init__(self, corpus_path):
		with open(corpus_path, 'rb') as pickle_in:
			corpus = pickle.load(pickle_in, encoding='utf8')

		# print(set(bigrams(corpus)))
		print(set(trigrams(corpus)))



def main():
	lm = LM('../bin/corpus.pkl')

if __name__ == "__main__":
   main()