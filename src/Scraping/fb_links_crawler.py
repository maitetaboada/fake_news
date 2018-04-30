from newspaper import Article
import pandas as pd


# continuation of the script written by helen.
def main():
    article_links = pd.DataFrame.from_csv("article_links_fbgraph.csv", encoding="ISO-8859-1")
    article_links_fb = pd.DataFrame.from_csv("facebook-fact-check.csv", encoding="ISO-8859-1")

    article_links_df = pd.DataFrame.merge(article_links, article_links_fb, on='ID.1', how='inner')
    article_text_df = pd.DataFrame(columns=['external_url', 'text', 'error'])

    for i, link in article_links_df.iterrows():
        article_text = {}
        try:
            article = Article(url=link['external_url'])
            article.download()
            article.parse()
            article_text['external_url'] = link['external_url']
            article_text['text'] = " ".join(article.text.split())
            if len(article_text['text']) > 32000:
                article_text['error'] = "Article too long"
            else:
                article_text['error'] = "No Error"
        except Exception as e:
            article_text['external_url'] = link['external_url']
            article_text['error'] = article.download_exception_msg or "Website down"
            article_text['text'] = "--"
        article_text_df = article_text_df.append(article_text, ignore_index=True)

    article_links_df = pd.DataFrame.merge(article_links_df, article_text_df, on='external_url', how='inner')
    article_links_df = article_links_df.drop_duplicates(keep='first')
    article_links_df = article_links_df.rename(index=str, columns={"ID.1": "ID", "external_url": "URL", "Rating": "label"})

    cols = ['ID', 'URL', 'label', 'text', 'error']
    article_links_df.to_csv('article_text.csv', index=False, columns=cols)


if __name__ == "__main__":
    main()