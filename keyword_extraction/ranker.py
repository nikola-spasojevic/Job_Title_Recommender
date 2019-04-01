from feature_extraction import TFIDF
import autocomplete
import pickle

class Ranker:

	@staticmethod
	def keyword_suggestions(bin, job_title):

		tf_idf = TFIDF(bin, job_title)
		# tf_idf.tfidf_vector=tfidf_transformer.transform(self.cv.transform([job_title]))

		# with open(bin+'words_freq.pkl', 'rb') as pickle_in:
		# 	words_freq = pickle.load(pickle_in, encoding='utf8')

		autocomplete.models.train_models(tf_idf.corpus)

		autocomplete.predict('software','eng')


def main():
	ranked = Ranker.keyword_suggestions('../bin/', 'manager')

if __name__ == "__main__":
   main()
