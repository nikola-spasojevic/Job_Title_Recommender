from trie import Trie
import pickle
from heapq import heappush, heappushpop, heappop

TOKEN_FREQ_DIR = '../bin/ngram_freq.pkl'

class Autocomplete:
	def __init__(self, token_freq_path):
		with open(token_freq_path, 'rb') as pickle_in:
			self.token_freq = pickle.load(pickle_in, encoding='utf8')

		self.trie = Trie()
		for token in self.token_freq:
			self.trie.insert(token)

	def search(self, prefix, capacity=5):
		min_heap = []
		trie_generator = self.trie.all_words_beginning_with_prefix(prefix)

		while True:
			try:
				val = next(trie_generator)
				if len(min_heap) < capacity:
					heappush(min_heap, (self.token_freq[val], val))
				else:
					heappushpop(min_heap, (self.token_freq[val], val))
				
			except StopIteration:
				break
	
		return sorted([heappop(min_heap)[1] for i in range(len(min_heap))])

def main():
	autocomplete = Autocomplete(TOKEN_FREQ_DIR)
	print(autocomplete.search('pfle'))

if __name__ == "__main__":
   main()