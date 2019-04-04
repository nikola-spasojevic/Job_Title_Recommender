import pickle

CORPUS_DIR = '../bin/corpus.pkl'

class Tokenizer:
	@staticmethod
	def job_data_tokenizer(corpus_dir=CORPUS_DIR):

		with open(corpus_dir, 'rb') as pickle_in:
			corpus = pickle.load(pickle_in, encoding='utf8')

		tokenized_corpus = []
		for i in corpus:
			tokenized_corpus.append(i.split())

		with open('../bin/tokenized_corpus.pkl', 'wb') as output:
			pickle.dump(tokenized_corpus, output)
			output.close()

def main():
	Tokenizer.job_data_tokenizer()

if __name__ == "__main__":
   main()