from newspaper import Article
import pandas as pd

# the page to be crawled needs to have its url in column "page_url"
def main(input_file, output_file):

    article_links = pd.DataFrame.from_csv(input_file, encoding="ISO-8859-1", index_col=None)
    article_links['ID'] = range(1, len(article_links) + 1)
    article_text_df = pd.DataFrame()

    for i, link in article_links.iterrows():

        article_text = {}
        try:
            url = link.page_url

            article = Article(url=url)
            article.download()
            article.parse()
            article_text['text'] = " ".join(article.text.split())
            if not article.text:
                article_text['error'] = "Resource moved"
                article_text['text'] = '--'

            elif len(article_text['text']) > 32000:
                article_text['error'] = "Article too long"
            elif any(x in url for x in ["twitter", "facebook", "youtu.be", "youtube", "reddit", "flickr", "wikisource"]):
                article_text['error'] = "Not a news article"
            else:
                article_text['error'] = "No Error"
        except Exception as e:
            article_text['error'] = article.download_exception_msg or "Website down"
            article_text['text'] = '--'

            print(e)
            print(url)

        article_text['ID'] = link['ID']
        article_text_df = article_text_df.append(article_text, ignore_index=True)

    article_links_final = pd.DataFrame.merge(article_links, article_text_df, on="ID", how="inner")
    article_links_final.to_csv(output_file, index=False)


if __name__ == "__main__":
    input_file = input("Please enter the input file name: ") #e.g. snopes.csv
    output_file = input("Please enter the output file name: ")
    main(input_file, output_file)