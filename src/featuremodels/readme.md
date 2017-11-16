# STS

## Notes on the feature-based method:
Easy:
1. Longest Common Substring
2. Longest Common Subsequence
3. Character n-grams n=1,2,3,4 
4. compare word n-grams using the Jaccard coefficient [Lyon et al. (2001)]
5. stopword n-grams (Stamatatos, 2011) n = 2, 3, . . . , 10
6. part-of-speech n-grams for various POS tags which we then compare using the containment measure and the Jaccard coefficient
7. word pair distance
8. function word frequencies measure (Dinu and Popescu, 2009) which operates on a set of 70 function words identified by Mosteller and Wallace (1964). Function word frequency vectors are computed and compared by Pearson correlation.
9. typetoken ratio (TTR) (Templin, 1957)

Intermediate:
1. Greedy String Tiling
2. Word pair order [Hatzivassiloglou et al., 1999]
3. sequential TTR (McCarthy and Jarvis, 2010).


Difficult:
1. Pairwise Word Similarity [Jiang and Conrath (1997), Lin (1998a) and Resnik (1995) on WordNet (Fellbaum, 1998)]
2. Explicit Semantic Analysis (ESA) [(Gabrilovich and Markovitch, 2007)]
3. Distributional Thesaurus (only the feature based on cardinal numbers(CD))
4. Lexical Substitution System
5. Statistical Machine Translation [Moses SMT system]

Experimental Setup:
1. pre-processing phase, we tokenize the input texts and lemmatize using the Tree-Tagger implementation (Schmid, 1994)
2. pre-computed similarity scores, and combines their log-transformed values using a linear regression classifier from the WEKA toolkit [Hall et al., 2009]
3. Post Processing: stripped all characters off the texts which are not in the character range [a-zA-Z0-9]. If the texts match, we set their similarity score to 5.0 regardless of the classifier’s output.
