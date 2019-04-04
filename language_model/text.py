# The below turns the n-gram-count dataframe into a Pandas series with the n-grams as indices for ease of working with the counts.  The second line can be used to limit the n-grams used to those with a count over a cutoff value.
sums = ngram_count.sum(axis = 0)
sums = sums[sums > 0]
ngrams = list(sums.index.values)
# The function below gives the total number of occurrences of 1-grams in order to calculate 1-gram frequencies
def number_of_onegrams(sums):
    onegrams = 0
    for ng in ngrams:
        ng_split = ng.split(" ")
        if len(ng_split) == 1:
            onegrams += sums[ng]
    return onegrams
# The function below makes a series of 1-gram frequencies.  This is the last resort of the back-off algorithm if the n-gram completion does not occur in the corpus with any of the prefix words.
def base_freq(og):
    freqs = pd.Series()
    for ng in ngrams:
        ng_split = ng.split(" ")
        if len(ng_split) == 1:
            freqs[ng] = sums[ng] / og
    return freqs
# For use in later functions so as not to re-calculate multiple times:
bf = base_freq(number_of_onegrams(sums))
# The function below finds any n-grams that are completions of a given prefix phrase with a specified number (could be zero) of words 'chopped' off the beginning.  For each, it calculates the count ratio of the completion to the (chopped) prefix, tabulating them in a series to be returned by the function.  If the number of chops equals the number of words in the prefix (i.e. all prefix words are chopped), the 1-gram base frequencies are returned.
def find_completion_scores(prefix, chops, factor = 0.4):
    cs = pd.Series()
    prefix_split = prefix.split(" ")
    l = len(prefix_split)
    prefix_split_chopped = prefix_split[chops:l]
    new_l = l - chops
    if new_l == 0:
        return factor**chops * bf
    prefix_chopped = ' '.join(prefix_split_chopped)
    for ng in ngrams:
        ng_split = ng.split(" ")
        if (len(ng_split) == new_l + 1) and (ng_split[0:new_l] == prefix_split_chopped):
            cs[ng_split[-1]] = factor**chops * sums[ng] / sums[prefix_chopped]
    return cs
# Example of completion scores:
find_completion_scores('in the national', 0, 0.4)