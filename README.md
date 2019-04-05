# Job_Title_Recommender

This is a project used to improve the effectiveness of writing Job Title postings.

# Autocompletion functionality implemented using Trie and Bag-Of-Words model
We first tokenize our text body and build a vocabulary of tokens (tokens can be unigrams, bigrams or trigrams - for this project).
The BoW model takes into account #ONLY the tokens' relative frequency in the ngram vocabulary (not their order or position).
In other words, no context is used to complete a word (unigram) or a couple of words (bigram).

This, combined with a Trie datastructure that simulates a graph of letter states, takes into account the input string and search all subtrees for possible word endings, along with their realtive word frequencies and returns a ranking of the top 5 results. 

Example
e.g. when inputting the word: 'jav', we can get a few possible results ranked by there relative unigram frequency scores:

job_title_recommender.auto_complete('jav')

	[(0.006204173716864072, 'java'), (0.00018800526414739614, 'javascript'), (0.00018800526414739614, 'javaee')]

We could expand this Relative Freuqeuncy Trie Scoring System to Bigrams and Trigrams, but as mentioned earlier, this kind of model does not account for context of previous words/inputs, but rather just how often this sequence appeared in our text.


# Keyword suggestion functionality is based of tri/bigram statistical language model using Maximum Likelihood Estimatation
We want to account for previous occurences of word sequences. Based upon Maximum Likelihood, a sequence of words (context) has a set of 'next word' results. This is used to predict the next possible words based on training data
This being MLE, the model returns a single word's relative frequency to the context (either 2 or 1 word prior to it) as its score.

Example
e.g. when inputting the word: 'java software', we can get a results ranked by there relative trigram frequency scores:

input: job_title_recommender.propose_next_word('java software')

	[(0.6, 'engineer'), (0.2, 'entwickler'), (0.2, 'developer')]


*if the second most previous word is out of context (or doesn't exist), we revert to a bigram model and return the MLE model's highest scoring bigram values

input: job_title_recommender.propose_next_word('hsbka software')

	[(0.3880597014925373, 'engineer'), (0.16417910447761194, 'entwickler'), (0.1044776119402985, 'ingenieur'), (0.07462686567164178, 'developer'), (0.04477611940298507, 'architect')]

*If the most previous word is out of context (or doesn't exist), we revert back to the autocmplete/unigram model.
Or, we can still use the MLE models unigram model score (the 2 models should be compared in further work)



	


Predict the probability of an equivalence class using its relative frequency in the training data:
– C(x) = count of x in training, N = number of training instances
● Problems with MLE:
– Underestimates the probability for unseen data: C (x)=0
● Maybe we just didn't have enough training data
– Overestimates the probability for rare data: C (x)=1
● Estimates based on one training sample are unreliable
– Solution: smoothing
