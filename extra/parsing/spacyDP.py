import spacy
import timeit
import numpy as np
from nltk import Tree
import pandas as pd





def test_dependency(file_name = "~/workspace/temp/grim/posts_fatemeh.csv" ):
    df = pd.read_csv(file_name, encoding="ISO-8859-1")

    my_string = df["body"][1]
    my_string = "Hello my dear! According to one of our reporters, the news was fake."
    my_string = my_string.replace(".", ".\n")
    print(my_string)
    doc = nlp(my_string)
    for span in doc.sents:
        print("#> span:", span)
    for word in doc:
        #print(word.dep_)
        if word.dep_ in ('ccomp','nmod', 'attr'):
            print("-------")
            subtree_span = doc[word.left_edge.i : word.right_edge.i + 1]
            print(word.dep_ , '|', subtree_span.text, '|', subtree_span.root.head.text)
            for child in subtree_span.root.head.children:
                if child.dep_ == 'nsubj':
                    print(child.text)
                    break
            print("-------")



#doc = nlp("The quick brown fox jumps over the lazy dog.")

#doc = nlp("She said she would not consider this case anymore.")

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

def tok_format(tok):
    return "_".join([tok.orth_, tok.tag_])

def to_nltk_tree_pos(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree_pos(child) for child in node.children])
    else:
        return tok_format(node)


[to_nltk_tree_pos(sent.root).pretty_print() for sent in doc.sents]


tic = timeit.default_timer()
nlp = spacy.load('en_core_web_sm')
#nlp = spacy.load('en_core_web_lg')

test_dependency()

toc = timeit.default_timer()
print(toc - tic)
