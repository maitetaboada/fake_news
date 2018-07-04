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

		print(e)
		print(url)
	return [error, text, title, publish_date, author]

def main(input_file, output_file):
	PAGE_FIRST_CITE_INDEX = -1
	ORIGIN_URL_INDEX = -2
	#filter invalid page out. we think the following formats are invalid page to use.
	INVALID_PAGE_SUFFIX = [".pdf", ".jpg", ".mp4", ".mp3", ".tif", '.tiff', '.gif', '.jpeg','.jif','jfif','.png', '.jpx']

	with open(input_file) as f:
		reader = csv.reader(f)
		# prepare and write header in
		title = next(reader)
		title += ['error_phase2', 'original_article_text_phase2', \
		'article_title_phase2', 'publish_date_phase2', 'author_phase2']
		title = ','.join(title) + '\n'
		with open(output_file, 'w', encoding="utf-8") as o:
			o.write(title)

		for line in reader:
			page_is_first_citation = (line[PAGE_FIRST_CITE_INDEX] == 'True')
			url = line[ORIGIN_URL_INDEX]

			not_invalid = True
			for i in INVALID_PAGE_SUFFIX:
				if i in url.lower():
					not_invalid = False

			if page_is_first_citation and not_invalid: # only parse the urls shows on 
										# first three paragraphs or first citation	
				original_article_info = get_origin_article_info(url) #pick the first url as the one
				#original_article_info = [str(l) for l in original_article_info] #"ISO-8859-1"?
				error = original_article_info[0]
				#if error == "No Error": 
				#Only parse the first page on source.
				new_line = line + original_article_info
				with open(output_file, 'a', newline='', encoding="utf-8") as o:
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
