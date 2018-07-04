import pandas as pd
import re

from urllib.parse import urlparse


def is_ascii(text):
    if isinstance(text, str):
        try:
            text.encode('ascii')
        except UnicodeEncodeError:
            return False
    else:
        try:
            text.decode('ascii')
        except UnicodeDecodeError:
            return False
    return True


def replace_bracket_text(str):
    word = re.findall("\(\w+\)", " ".join(str.split()[:10]))
    word2 = re.findall("\[\w+\]", " ".join(str.split()[:10]))

    if word:
        str = str.replace(word[0], "")
    if word2:
        str = str.replace(word2[0], "")
    return str


def replace_question_marks(str):
    words = re.findall("\?\w+\? | \w+\?\w+", str)

    for word in words:
        str = str.replace(word, re.findall("\w+", word)[0])

    return str


def main(input_file, output_file):
    unwanted_text = ['This Account has been suspended', 'Not Found The requested URL',
                     'Tap here to turn on desktop notification', 'This website uses cookie', 'Please Sign In and use',
                     'Log in to access content', 'External links are provided for reference purposes',
                     'Page not found', 'Here are some suggestions for finding', "Some Word features can't be displayed",
                     'Small (S) has the shortest download time', 'Permission (Reusing this file)', 'browser must have JavaScript',
                     'Log In Logging in for the first time', '404 - Page Not Found', '404 Error', 'You are free: to share',
                     'Copyright. All Rights Reserved', 'Some older articles might not yet have been migrated',
                     'Please enable Cookies on your browser', 'Enter your first name: Enter your last name:',
                     'Need Help? Chat Now', 'Close Get email notifications', 'Comp licence:',
                     'Donations are welcome at the time of your visit', "Image searched for: You've reached your daily search limit",
                     'Read, highlight, and take notes, across web, tablet, and phone', 'Send me email updates and offers from Fox',
                     'I, the copyright holder of this work, hereby publish it under the following license',
                     'Just One More Thing... We have sent you a verification email.', 'Please upgrade your browser',
                     'Tor Tor is an encrypted anonymising network that makes it harder', 'Note: Javascript is disabled',
                     'LII has no control over and does not endorse any', 'TinEye is an image search and recognition company',
                     'search textbox has an autosuggest feature', 'interactive transcript could not be loaded', 'Page Not Found',
                     ]

    df = pd.read_csv(input_file, encoding="ISO-8859-1")
    df = df[df.error == 'No Error']

    df_w_dups = len(df)
    df = df.drop_duplicates(keep='first', subset=['claim_label', 'text'])
    df_actual_len = len(df)

    df['page_domain'] = df['page_url'].apply(lambda x: urlparse(x).netloc)

    df_filter_short = df[df['original_article_text_phase2'].apply(lambda x: len(x.split()) > 40)]
    df_filter_long = df_filter_short[df_filter_short['original_article_text_phase2'].apply(lambda x: len(x) < 30000)]

    short_len_text = len(df) - len(df_filter_short)
    long_len_text = len(df_filter_short) - len(df_filter_long)

    df_cleaned = df_filter_long[~df_filter_long['original_article_text_phase2'].apply(lambda x: any(t in x for t in unwanted_text))]
    unwanted_text_len = len(df_filter_long) - len(df_cleaned)
    df_cleaned['text'] = df_cleaned['text'].apply(lambda x: replace_bracket_text(x))

    df_cleaned['is_ascii'] = df_cleaned['text'].apply(lambda x: is_ascii(x))

    """FOR SNOPE"""
    df_cleaned['text'] = df_cleaned['text'].apply(lambda x: x.encode("ascii", errors='ignore').decode()) # remove non ascii characters

    """ FOR EMERGENT, POLITIFACT AND BUZZFEED"""
    df_cleaned['text'] = df_cleaned['text'].apply(lambda x: replace_question_marks(x))

    df_not_ascii = len(df_cleaned[df_cleaned['is_ascii'] == False])
    df_cleaned.to_csv(output_file, index=False)

    statistics = {}
    statistics['len_with_duplicates'] = df_w_dups
    statistics['len_without_duplicates'] = df_actual_len
    statistics['num_of_duplicate_rows'] = df_w_dups - df_actual_len
    statistics['number_of_unwanted_text'] = unwanted_text_len
    statistics['number_of_short_text'] = short_len_text
    statistics['number_of_long_text'] = long_len_text
    statistics['number_of_non_ascii_text'] = df_not_ascii

    stats_df = pd.DataFrame.from_dict(statistics, orient='index')
    stats_df.to_csv('statistics.csv')


if __name__ == "__main__":
    input_file = input("Please enter input file name: ")
    output_file = input("Please enter output file name: ")
    main(input_file, output_file)