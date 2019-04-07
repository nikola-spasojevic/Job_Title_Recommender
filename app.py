from src.text_preprocessing import TextPreprocessing
from src.bag_of_words import BagOfWords, TRAIN_CORPUS_DIR
from src.language_model import NgramLanguageModel, NGRAMS_DIR
from src.job_title_recommender import JobTitleRecommender, LIKELIHOODS_DIR, UNIGRAM_FREQ_DIR
import sys
import os

def main():
	if not os.path.isfile(TRAIN_CORPUS_DIR):
		TextPreprocessing.corpus_gen()
	if not os.path.isfile(UNIGRAM_FREQ_DIR) or not os.path.isfile(NGRAMS_DIR):
		BagOfWords.ngram_frequencies_gen()
	if not os.path.isfile(LIKELIHOODS_DIR):
		NgramLanguageModel.likelihoods_gen()

	job_title_recommender = JobTitleRecommender()
	file = open('bin/model_evaluation.txt', 'r')
	print (file.read())
	
	print("Input Job Title: ")
	while True:		
		var = input('\n')
		result_list = []

		if var:
			if var[-1] == ' ':
				result_list.extend(job_title_recommender.predict_next_word(var[:-1]))
				print('\n{:<20}{:>10}'.format('Next Word Prediction', 'Score'))
			else:
				result_list.extend(job_title_recommender.auto_complete(var))
				print('\n{:<20}{:>10}'.format('Autocmplete', 'Score'))
			
			for score, word in result_list:
				print('{:<20} |{:>1.10f}'.format(word, score))

if __name__ == "__main__":
   main()