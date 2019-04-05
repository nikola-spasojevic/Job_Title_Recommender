from src.text_preprocessing import TextPreprocessing
from src.bag_of_words import BagOfWords, CORPUS_DIR
from src.language_model import NgramLanguageModel, NGRAMS_DIR
from src.job_title_recommender import JobTitleRecommender, LIKELIHOODS_DIR, UNIGRAM_FREQ_DIR
import os

def main():
	if not os.path.isfile(CORPUS_DIR):
		TextPreprocessing.corpus_gen()
	if not os.path.isfile(UNIGRAM_FREQ_DIR) or not os.path.isfile(NGRAMS_DIR):
		BagOfWords.ngram_frequencies_gen()
	if not os.path.isfile(LIKELIHOODS_DIR):
		NgramLanguageModel.calc_likelihood()
	
	job_title_recommender = JobTitleRecommender()

	print(job_title_recommender.auto_complete('jav'))
	print(job_title_recommender.propose_next_word('gibberish software'))


	# Evaluation to be done using word error rate (WER. using edit distance
if __name__ == "__main__":
   main()