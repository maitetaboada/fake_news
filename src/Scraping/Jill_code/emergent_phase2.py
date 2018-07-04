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
	text = ''
	error = ''
	publish_date = ''
	author = ''
	title = ''
	try:
		if len(url) > 0:
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

		print(e)
		print(url)
	return [error, text, title, publish_date, author]

def main(input_file, output_file):
	with open(input_file) as f:
		reader = csv.reader(f)
		title = next(reader)
		title += ['error_phase2', 'original_article_text_phase2', \
		'article_title_phase2', 'publish_date_phase2', 'author_phase2']
		title = ','.join(title) + '\n'
		with open(output_file, 'w', encoding="utf8") as o:
			o.write(title)
		for line in reader:
			url = line[-1]
			#url = ast.literal_eval(url) # convert to url list
			original_article_info = get_origin_article_info(url) #pick the first url as the one
			#original_article_info = [str(l) for l in original_article_info] #"ISO-8859-1"?
			new_line = line + original_article_info
			with open(output_file, 'a', newline='', encoding='utf8') as o:
				csv_writer = csv.writer(o)
				csv_writer.writerow(new_line)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='fetching the possible existing article information')
    parser.add_argument('input_file', type=str,
                    help='the input file')
    parser.add_argument('output_file', type=str,
                    help='the output file')
    args = parser.parse_args()
    main(args.input_file, args.output_file)


