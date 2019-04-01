#%matplotlib inline
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import pickle
from text_preprocessing import stop_words

class WordCloudPlot:

	@staticmethod
	def get_figure(corpus_pickle):
		corpus = pickle.load(open(corpus_pickle, 'rb'))

		wordcloud = WordCloud(
		                          background_color='white',
		                          stopwords=stop_words,
		                          max_words=100,
		                          max_font_size=50, 
		                          random_state=42
		                         ).generate(str(corpus))
		print(wordcloud)
		fig = plt.figure(1)
		plt.imshow(wordcloud)
		plt.axis('off')
		plt.show()
		fig.savefig('word1.png', dpi=900)

def main():
	WordCloudPlot.get_figure('corpus.pkl')

if __name__ == "__main__":
   main()
