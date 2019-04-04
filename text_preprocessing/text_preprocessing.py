import numpy as np
import pandas as pd
import pickle
import re
import nltk
import codecs
#nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
#nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer

JOB_TITLE_DATA_DIR = '../input/jobcloud_published_job_titles.csv'

stop_words = set(stopwords.words('english'))
stop_words_french = set(stopwords.words('french'))
stop_words = stop_words.union(stop_words_french)
stop_words_german = set(stopwords.words('german'))
stop_words = stop_words.union(stop_words_german)

# Process the iput data to get a valid body of text
class TextPreprocessing:
    @staticmethod
    def corpus_gen(job_title_data_dir=JOB_TITLE_DATA_DIR):
        dataset = pd.read_csv(job_title_data_dir, header=None, names=['Job Titles'], encoding='utf8')
        
        corpus, tokenized_corpus = [], []
        for i in range(len(dataset)):
            #Convert to lowercase
            text = dataset['Job Titles'][i].lower()
            
            #remove tags
            text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

            # remove special characters and digits
            text=re.sub("(\\d|\\W)+"," ",text)

            #Convert to list from string
            text = text.split()

            #remove all stopwords from text body
            text = [word for word in text if word not in stop_words]

            # The goal of both stemming and lemmatization is to reduce inflectional forms and sometimes derivationally 
            # related forms of a word to a common base form. (e.g beginning -> begin, cars -> car) 
            
            #Lemmatisation
            lem = WordNetLemmatizer()
            text = [lem.lemmatize(word) for word in text]

            #Stemming - would improve the quality of proposed results
            # porter = PorterStemmer()
            # text = [porter.stem(word) for word in text if not word in stop_words]

            text = " ".join(text)
            corpus.append(text)

        with open('../bin/corpus.pkl', 'wb') as output:
            pickle.dump(corpus, output)
            output.close()

def main():
    TextPreprocessing.corpus_gen()

if __name__ == "__main__":
   main()
