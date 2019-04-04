from .helper.trie import Trie
import pickle
from heapq import heappush, heappushpop, heappop
from collections import deque

UNIGRAM_FREQ_DIR='bin/unigram_freq.pkl'
LIKELIHOODS_DIR='bin/likelihoods.pkl'
CAPACITY=5
N_GRAM=3

class JobTitleRecommender:
	def __init__(self, unigram_freq_dir=UNIGRAM_FREQ_DIR, likelihood_dir=LIKELIHOODS_DIR):
		with open(unigram_freq_dir, 'rb') as pickle_in:
			self.unigram_freq = pickle.load(pickle_in, encoding='utf8')

		with open(likelihood_dir, 'rb') as pickle_in:
			self.ngram_likelihood = pickle.load(pickle_in, encoding='utf8')

		# print(self.bi_gram_likelihood)

		self.trie = Trie()
		for unigram in self.unigram_freq:
			self.trie.insert(unigram)

	# Retrive top CAPACITY results based upon unigram frequencies 
	# Frequencies are calculated in the Bag Of Words module and represent occurence of all unigrams
	# This is only useful for single words, since BoW does not account for context of previous text
	def auto_complete(self, prefix, capacity=CAPACITY):
		min_heap = []
		trie_generator = self.trie.all_words_beginning_with_prefix(prefix)
		while True:
			try:
				val = next(trie_generator)
				if len(min_heap) < capacity:
					heappush(min_heap, (self.unigram_freq[val], val))
				else:
					heappushpop(min_heap, (self.unigram_freq[val], val))
				
			except StopIteration:
				break
	
		return sorted([heappop(min_heap) for i in range(len(min_heap))], reverse=True)

	# Retrive top CAPACITY results based upon the unigram frequencies and previous set of (up to N_GRAM) words
	# likelihoods are precalculated in the Language Model module
	# Based upon Maximum Likelihood, a sequence of words (context) has a set of 'next word' results.
	# This is used to predict the next possible words based on training data# This is only useful for single words, since BoW does not account for context of previous text
	def propose_next_word(self, prev_text, capacity=CAPACITY):
		min_heap = []
		prev_text_tokens = deque(prev_text.split(' ')[-N_GRAM+1:])

		# if current n_gram model does not work with prev_text, lower n_gram module by 1 until 
		# either valid model is found or all ngram models are exhausted
		while prev_text_tokens and tuple(prev_text_tokens) not in self.ngram_likelihood:
			prev_text_tokens.popleft()			

		if prev_text_tokens:
			t = tuple(prev_text_tokens)
			if t in self.ngram_likelihood:
				for x in self.ngram_likelihood[t]:
					if len(min_heap) < capacity:
						heappush(min_heap, x)
					else:
						heappushpop(min_heap, x)

		return sorted([heappop(min_heap) for i in range(len(min_heap))], reverse=True)

def main():
	job_title_recommender = JobTitleRecommender()
	print(job_title_recommender.auto_complete('engin'))
	print(job_title_recommender.propose_next_word('ajksd ajskdk jkasdn java engineering'))

if __name__ == "__main__":
   main()