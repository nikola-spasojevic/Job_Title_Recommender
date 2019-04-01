import numpy as np
import pandas as pd
import pickle
import re
import nltk
import codecs
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
stop_words_french = set(stopwords.words('french'))
stop_words = stop_words.union(stop_words_french)
stop_words_german = set(stopwords.words('german'))
stop_words = stop_words.union(stop_words_german)
# new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]
# stop_words = stop_words.union(new_words)

class TextPreprocessing:

    @staticmethod
    def get_corpus(file_path):
        # dataset = pd.read_csv('./nlp_assignment/jobcloud_published_job_titles.csv', header=None, names=['Job Titles'])
        dataset = pd.read_csv(file_path, header=None, names=['Job Titles'], encoding='utf8')
        freq = pd.Series(' '.join(dataset['Job Titles']).split()).value_counts()[:300]        
        
        corpus = []
        for i in range(len(dataset)):
            #Convert to lowercase
            text = dataset['Job Titles'][i].lower()
            
            #remove tags
            text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

            # remove special characters and digits
            text=re.sub("(\\d|\\W)+"," ",text)

            ##Convert to list from string
            text = text.split()

            ##Stemming
            ps=PorterStemmer()

            #Lemmatisation
            lem = WordNetLemmatizer()
            text = [lem.lemmatize(word) for word in text if not word in stop_words] 
            text = " ".join(text)
            corpus.append(text)

        with open('../bin/corpus.pkl', 'wb') as output:
            pickle.dump(corpus, output)
            output.close()  

def main():
    TextPreprocessing.get_corpus('../input/jobcloud_published_job_titles.csv')

if __name__ == "__main__":
   main()
