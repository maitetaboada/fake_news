from newspaper import Article
import time
import ast
import csv

def get_origin_article_info(url):
	"""get all the fact tag from the webpage https://www.snopes.com/archive/ ONLY

	TODO: add a validation for the accessiability of the webpage

	Args:
		correct URL link

	Return:
		a list of all kind of article information
	"""
	INVALID_PAGE_SUFFIX = [".pdf", ".jpg", ".mp4", ".mp3", \
	".tif", '.tiff', '.gif', '.jpeg','.jif','jfif','.png', '.jpx']
	text = ''
	error = ''
	publish_date = ''
	author = ''
	title = ''
	valid = True
	for i in INVALID_PAGE_SUFFIX:
		if i in url.lower():
			valid = False
	if valid:
		try:
			time.sleep(1)
			print(url)
			article = Article(url=url)
			article.download()
			article.parse()
			text = " ".join(article.text.split())
			
			if not article.text:
				error = "Resource moved"
				text = '--'

			elif len(text) > 32000:
				error = "Article too long"

			elif len(text.split(" ")) < 50: # if the text contains less than 50 words, we think it is too short
				error = "Article too short"

			elif any(x in url for x in ["twitter", "facebook", "youtu.be", "youtube", "reddit", "flickr", "wikisource"]):
				error = "Not a news article"

			else:
				error = "No Error"
				if article.publish_date:
					publish_date = article.publish_date.strftime('%Y-%m-%d')

				if article.authors:
					author = article.authors[0] # ONLY kept the first author

				if article.title:
					title = article.title

		except Exception as e:
			error = article.download_exception_msg or "Website down"
			text = '--'
	return [error, text, title, publish_date, author]
