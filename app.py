from src.text_preprocessing import TextPreprocessing
from src.bag_of_words import BagOfWords
from src.language_model import NgramLanguageModel
from src.job_title_recommender import JobTitleRecommender

def main():
	TextPreprocessing.corpus_gen()
	BagOfWords.ngram_frequencies_gen()
	NgramLanguageModel.calc_likelihood()
	job_title_recommender = JobTitleRecommender()
	print(job_title_recommender.auto_complete('engin'))
	print(job_title_recommender.propose_next_word('ajksd ajskdk jkasdn java engineering'))

if __name__ == "__main__":
   main()