from trie import Trie
import pickle
from heapq import heappush, heappushpop, heappop

class Autocomplete:
	def __init__(self, word_freq_path):
		with open(word_freq_path, 'rb') as pickle_in:
			self.word_freq = pickle.load(pickle_in, encoding='utf8')

		self.trie = Trie()
		for word in self.word_freq:
			self.trie.insert(word)

	def search(self, prefix, capacity=5):
		min_heap = []
		x = self.trie.all_words_beginning_with_prefix(prefix)

		while True:
			try:
				val = next(x)
				if len(min_heap) < capacity:
					heappush(min_heap, (self.word_freq[val], val))
				else:
					heappushpop(min_heap, (self.word_freq[val], val))
				
			except StopIteration:
				break
	
		return sorted([heappop(min_heap)[1] for i in range(len(min_heap))])

def main():
	autocomplete = Autocomplete('../bin/word_freq.pkl')
	print(autocomplete.search('pflegefachfrau'))

if __name__ == "__main__":
   main()