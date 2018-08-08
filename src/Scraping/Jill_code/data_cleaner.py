# -*- coding: utf-8 -*-
import re
import csv
import pandas as pd
from urllib.parse import urlparse
from more_itertools import unique_everseen

def _clean_word(word):
    """
    Filter out the charater in words which is not alphabetic
    """
    return ''.join(letter for letter in word.lower() if 'a' <= letter <= 'z')

def _count_words(line):
    """
    count the number of real words in a line
    """
    count = 0
    for i in line.split():
        if _clean_word(i):
            count += 1
    return count

def keep_suitable_length_article(line):
    """ 
    keep the article with length more than 50 words
     and less than 30000 characters
    """
    if _count_words(line) >= 100 and _count_words(line) < 30000:
        return line
    else:
        return ''

def replace_unusual_characters(line):
    """replace unusual characters in the word
        1. replace wired "’" apostrophe
        2. replace unusual quotation.
        3. ...

    """
    l = re.sub("’", "'", line)
    l = re.sub("‘", "'", l)
    l = re.sub("“", "\"", l)
    l = re.sub("”", "\"", l)
    l = re.sub("，", ",", l)

    return l

def replace_email_address(line):
    """replace email address as _EMAIL_
        e.g.:<robetded@ix.netcom.com> -> _EMAIL_
    """
    return re.sub(r'[\w\.-]+@[\w\.-]+', "_EMAIL_", line)

def replace_http_address(line):
    """replace email address as _EMAIL_
        e.g.:<robetded@ix.netcom.com> -> _EMAIL_
    """
    links = re.findall(r"http\S+", line)
    end = ""
    for link in links:
        if link[-1] == "." or link[-1] == ",":
         end = link[-1]
        line = line.replace(link, "_LINKS_" + end)

    return line

def is_unwanted_text(line):
    """ the text listed on the list are not wanted
    Args:
        line: String    The article
    Rturn:
        Bool    if true, we don't need this article, otherwise, we need it
    """
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
    for t in unwanted_text:
        if t in line:
            return True
    return False

def replace_question_marks(str):
    words = re.findall("\?\w+\? | \w+\?\w+", str)

    for word in words:
        str = str.replace(word, re.findall("\w+", word)[0])

    return str

def replace_bracket_text(str):
    word = re.findall("\(\w+\)", " ".join(str.split()[:10]))
    word2 = re.findall("\[\w+\]", " ".join(str.split()[:10]))

    if word:
        str = str.replace(word[0], "")
    if word2:
        str = str.replace(word2[0], "")
    return str

def reserve_correct_ascii(line):
    """reserve the correct ascii characters
    """
    return line.encode("ascii", errors='ignore').decode()

def is_debunking_websites(origin_web):
    DEBUNKING_WEBSITE = ["snope", "politifact"]
    for w in DEBUNKING_WEBSITE:
        if w in origin_web:
            return True
    return False

def clean_claim_example(line):
    return re.sub(r"SeeExample\( s \)", "", line).strip()

def clean_text(text):
    # data clean
    new_line = reserve_correct_ascii(text)
    new_line = replace_unusual_characters(new_line)
    new_line = replace_email_address(new_line)
    new_line = replace_http_address(new_line)
    new_line = replace_question_marks(new_line)
    new_line = replace_bracket_text(new_line)
    return new_line


def main(input_file, output_file, website_name):
    if website_name == "Snopes":
        TITLE_INDEX = 2
        CLAIM_INDEX = 5
        ARTICLE_TITLE_INDEX = -3
        TEXT_INDEX = -4
        ERROR_INDEX = -5
        ORI_URL_IDX = 6
    elif website_name == "politifact":
        TITLE_INDEX = 2
        CLAIM_INDEX = 3
        CLAIM_CITE_INDEX = 4
        ARTICLE_TITLE_INDEX = -3
        TEXT_INDEX = -4
        ERROR_INDEX = -5
        ORI_URL_IDX = 9
    elif website_name == "emergent":
        TITLE_INDEX = 3
        CLAIM_INDEX = 4
        ARTICLE_TITLE_INDEX = -3
        TEXT_INDEX = -4
        ERROR_INDEX = -5
        ORI_URL_IDX = -6

    with open(input_file) as f:
        reader = csv.reader(f)
        header = next(reader)
        with open(output_file, 'w', encoding="utf8") as o:
            o.write(",".join(header) + "\n")
        for l in reader:
            if l[ERROR_INDEX] == "No Error" and not is_debunking_websites(l[ORI_URL_IDX]):
                new_line = ""
                new_line = l[TEXT_INDEX]
                new_line = keep_suitable_length_article(new_line)
                if new_line and not is_unwanted_text(new_line):
                    l[TITLE_INDEX] = clean_text(l[TITLE_INDEX])
                    l[CLAIM_INDEX] = clean_text(l[CLAIM_INDEX])
                    l[CLAIM_INDEX] = clean_claim_example(l[CLAIM_INDEX])
                    if website_name == "politifact":
                        l[CLAIM_CITE_INDEX] = clean_text(l[CLAIM_INDEX])
                    l[ARTICLE_TITLE_INDEX] = clean_text(l[ARTICLE_TITLE_INDEX])
                    l[TEXT_INDEX] = clean_text(new_line)

                    with open(output_file, 'a', newline='', encoding='utf-8') as o:
                        csv_writer = csv.writer(o)
                        csv_writer.writerow(l)

    # remove duplicate
    #df = pd.read_csv(output_file)
    #df.drop_duplicates(keep='first', \
     #   subset=['original_article_text_phase2']).to_csv(output_file, index=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='clean the data')
    parser.add_argument('input_file', type=str, help='name of input file')
    parser.add_argument('output_file', type=str, help='name of output file')
    parser.add_argument('website_name', type=str, help='name of website')
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.website_name)