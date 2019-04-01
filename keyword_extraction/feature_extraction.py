from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix
from tokenization import Tokenizer
import pickle

class TFIDF:
	def __init__(self, bin, job_title):

		with open(bin+'normalized.pkl', 'rb') as pickle_in:
			self.normalized = pickle.load(pickle_in, encoding='utf8')

		with open(bin+'count_vectorizer.pkl', 'rb') as pickle_in:
			self.cv = pickle.load(pickle_in, encoding='utf8')

		with open(bin+'corpus.pkl', 'rb') as pickle_in:
			self.corpus = pickle.load(pickle_in, encoding='utf8')

		tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
		tfidf_transformer.fit(self.normalized)
		# get feature names
		self.feature_names=self.cv.get_feature_names()
	 
		# fetch document for which keywords needs to be extracted
		# self.doc=self.corpus[400]
		self.doc=job_title

		# print(self.feature_names)

		# print(cv.vocabulary_.keys())
 
		#generate tf-idf for the given document
		self.tfidf_vector=tfidf_transformer.transform(self.cv.transform([self.doc]))

		# print(self.tfidf_vector)

	"""get the feature names and tf-idf score of top n items""" 
	def extract_topn_from_vector(self, topn=10):
		#Function for sorting tf_idf in descending order
		def sort_coo(coo_matrix):
			tuples = zip(coo_matrix.col, coo_matrix.data)
			return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

		#sort the tf-idf vectors by descending order of scores
		sorted_items=sort_coo(self.tfidf_vector.tocoo())

		#use only topn items from vector
		sorted_items = sorted_items[:topn]
		score_vals = []
		feature_vals = []
		
		# word index and corresponding tf-idf score
		for idx, score in sorted_items:
			#keep track of feature name and its corresponding score
			score_vals.append(round(score, 3))
			feature_vals.append(self.feature_names[idx])

		#create a tuples of feature,score
		#results = zip(feature_vals,score_vals)
		results= {}
		for idx in range(len(feature_vals)):
			results[feature_vals[idx]]=score_vals[idx]

		return results

def main():
	tf_idf = TFIDF('../bin/', 'software engineer')

	#extract only the top n; n here is 10
	keywords=tf_idf.extract_topn_from_vector()
	 
	# now print the results
	print("\nAbstract:")
	print(tf_idf.doc)
	print("\nKeywords:")
	for k in keywords:
	    print(k,keywords[k])

if __name__ == "__main__":
   main()
