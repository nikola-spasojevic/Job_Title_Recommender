# Job Title Recommender

This is a project used to improve the effectiveness of writing Job Title postings.

In each step of the pipeline process, the intermittent state is saved to local memory and loaded into each next stage in order to maintain efficiency when prototyping models and tweaking.

**HOW TO RUN:**

Pull into local directory and run (make sure to have all relevant python3 libraries installed):
	
	python3 app.py
# Implemetation

## 1. Data Preparation
The First step is to clean the data and generate a valid corpus for training(95%) and testing(5% of total corpus).
This is done by removing special characters and digits, converting everything to lower case, removing stopwords from all languages, lemmatisation (and possibly stemming for future models), tokenization.

## 2. Get Model Vocabulary and Ngram frequencies
First of all, in the Bag of Words model, using the CountVectorizer class, we are able to extract all unigram, bigrams and trigrams from our training set and generate a frequency distribution for these ngrams.

The ngram frequencies are calculated in a dedicated dictionary, where the keys represent our entire ngram vocabulary.

## 3. Maximum Likelihood Estimation Language Model

The training data is first tokenized and padded in order to account for beggining and ending context. The data is then fitted into the MLE model which is used to calculate the likelihood of each token in each sentence given its previous context.

The purpose of training a language model is to have a score of how probable words are in certain contexts:

- If there is not context (i.e. a unigram model), the model returns the item's relative frequency as its score.
- But of there is a single word of context, we get a bigram model score.
- Similarly with two words of context, we get a trigram model score.

These likelihood scores are calculated and saved in memory in order to provide the Keyword suggestion functionality. Since the MLE may overfit the data: it will assign 0 probabilities to words it hasn't seen. We need smoothing:
http://mlwiki.org/index.php/Smoothing_for_Language_Models

	Usage of MLE: Predict the probability of an equivalence class using its relative frequency in the training data:

	– C(x) = count of x in training, N = number of training instances

	● Problems with MLE:

	– Underestimates the probability for unseen data: C (x)=0

	● Maybe we just didn't have enough training data

	– Overestimates the probability for rare data: C (x)=1

	● Estimates based on one training sample are unreliable

	– Solution: smoothing (In our model, we choose Laplace/Additive Smoothing)

## 4. Model Evaluation

A model is evaluated by calculating the total entropy of the language model on the test corpus. 
The entropy is calculated as the total sum of the log probability of each word given ngrams in the test corpus.
	
	Model Evaluation Score (Entropy): 12.333434915284093


# JobTitleRecommender Function Descriptions:



## Autocompletion functionality implemented using Trie and Bag-Of-Words model

We first tokenize our text body and build a vocabulary of tokens (tokens can be unigrams, bigrams or trigrams - for this project).
The BoW model takes into account #ONLY the tokens' relative frequency in the ngram vocabulary (not their order or position).
In other words, no context is used to complete a word (unigram) or a couple of words (bigram).

This, combined with a Trie datastructure that simulates a graph of letter states, takes into account the input string and search all subtrees for possible word endings, along with their realtive word frequencies and returns a ranking of the top 5 results. 

**Example**
e.g. when inputting the word: 'jav', we can get a few possible results ranked by there relative unigram frequency scores:

	input: job_title_recommender.auto_complete('jav')**

	[(0.006204173716864072, 'java'), (0.00018800526414739614, 'javascript'), (0.00018800526414739614, 'javaee')]

We could expand this Relative Frequency Trie Scoring System to Bigrams and Trigrams, but as mentioned earlier, this kind of model does not account for context of previous words/inputs, but rather just how often this sequence appeared in our text.

## Keyword suggestion functionality is based of tri/bigram statistical language model using Maximum Likelihood Estimation
We want to account for previous occurences of word sequences. Based upon Maximum Likelihood, a sequence of words (context) has a set of 'next word' results. This is used to predict the next possible words based on training data.
This being MLE, the model returns a single word's relative frequency to the context (either 2 or 1 word prior to it) as its score.

**Example**
e.g. when inputting the word: 'java software', we can get a results ranked by there relative trigram frequency scores:

	input: job_title_recommender.propose_next_word('java software')**

	[(0.6, 'engineer'), (0.2, 'entwickler'), (0.2, 'developer')]


*if the second most previous word is out of context (or doesn't exist), we revert to a bigram model and return the MLE model's highest scoring bigram values

	input: job_title_recommender.propose_next_word('gibberish software')**

	[(0.3880597014925373, 'engineer'), (0.16417910447761194, 'entwickler'), (0.1044776119402985, 'ingenieur'), (0.07462686567164178, 'developer'), (0.04477611940298507, 'architect')]

*If the most previous word is out of context (or doesn't exist), we revert back to the autocmplete/unigram model.
Or, we can still use the MLE models unigram model score (the 2 models should be compared in further work)

**TODO: Model Evaluation is to be done using Word Error Rate (WER)**
The WER score can be derived from the Edit Distance, which can be ran on teh test set in which we can see for each job posting how does the model complete the job posting given the data.

● Weighted scores can be given based on Insertion, Deletion, Substitution
