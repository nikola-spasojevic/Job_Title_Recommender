# Job_Title_Recommender

This is project used to improve the text of job postings

# Autocompletion functionality implemented using Trie and Bag-Of-Words model
We first tokenize our text body and build a vocabulary of tokens (tokens can be unigrams, bigrams or trigrams - for this project).
The BoW model takes into account #ONLY the tokens' relative frequency in the ngram vocabulary (not their order or position).
In other words, no context is used to complete a word (unigram) or a couple of words (bigram).

This, combined with a Trie datastructure that simulates a graph of letter states, takes into account the input string and search all subtrees for possible word endings, along with their realtive word frequencies and returns a ranking of the top 5 results. 

Example
e.g. when inputting the word: 'jav', we can get a few possible results ranked by there relative unigram frequency score:

	[(0.006204173716864072, 'java'), (0.00018800526414739614, 'javascript'), (0.00018800526414739614, 'javaee')]

We could expand this Relative Freuqeuncy Trie Scoring System to Bigrams and Trigrams, but as mentioned earlier, this kind of model does not account for context of previous words/inputs, but rather just how often this sequence appeared in our text.


# Keyword suggestion functionality is based of tri/bigram statistical language model using Maximum Likelihood Estimatation
We want to account for previous 
# Likelihood estimator for generic ngrams (excluding unigrams)
# likelihoods are precalculated in the Language Model module
	# Based upon Maximum Likelihood, a sequence of words (context) has a set of 'next word' results.
	# This is used to predict the next possible words based on training data# This is only useful for single words, since BoW does not account for context of previous text
	# This being MLE, the model returns a single word's relative frequency as its score. 
	# print(lm.unmasked_score('java')) P('java')
