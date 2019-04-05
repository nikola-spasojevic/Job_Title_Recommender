import pickle

TRAIN_CORPUS_DIR = 'bin/train_corpus.pkl'

class Tokenizer:
	@staticmethod
	def job_data_tokenizer(train_corpus_dir=TRAIN_CORPUS_DIR):
		with open(train_corpus_dir, 'rb') as pickle_in:
			train_corpus = pickle.load(pickle_in, encoding='utf8')

		tokenized_train_corpus = []
		for i in train_corpus:
			tokenized_train_corpus.append(i.split())

		return tokenized_train_corpus

def main():
	Tokenizer.job_data_tokenizer()

if __name__ == "__main__":
   main()