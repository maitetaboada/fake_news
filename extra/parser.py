import os
import pandas as pd
import multiprocessing as mp
import numpy as np
import pickle
from timeit import default_timer as timer
from multiprocessing import cpu_count
from nltk.parse import stanford
from nltk import Tree


os.environ['STANFORD_PARSER'] = './jars'
os.environ['STANFORD_MODELS'] = './jars'
path = './jars/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz'
parser = stanford.StanfordParser(model_path=path)


def parallelize(data, func):
    cores = cpu_count()
    partitions = cores
    data_split = np.array_split(data, partitions)
    pool = mp.Pool(cores)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def build_rules(article):
    rules = []
    text = article.split(".")
    for idx, sent in enumerate(text):
        try:
            tree = list(parser.raw_parse(sent))
            t = Tree.fromstring(str(tree[0]))
            rules += t.productions()
        except:
            continue
    return rules


def parse_article(df):
    df['rules'] = df['text'].apply(build_rules)
    return df


def main():

    file_name = input("Please enter the input file name: ")
    output_file = input("Please enter the output file name: ")

    # file_name = "./Sample2/testRaw"
    # output_file = "./Sample2/rules_testRaw2.csv"

    texts_valid = np.load(file_name)
    print(texts_valid)
    comments_df = pd.DataFrame(texts_valid, columns=["text"])
    start = timer()
    print("start: ", start)
    df_processed = parallelize(comments_df, parse_article)
    df_processed.to_csv(output_file, index=False)
    end = timer()

    print('Total time taken: ', end - start)


if __name__ == "__main__":
    main()


