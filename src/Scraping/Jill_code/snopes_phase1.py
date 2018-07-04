from bs4 import BeautifulSoup
import requests
import os
import re
import time
#import dbm
import csv

def basic_clean(sentence):
	return re.sub(u"\t", "", sentence)

def get_page(url):
	print(url)
	page = ''
	while page == '':
		try:
			time.sleep(1)
			page = requests.get(url)
		except:
			print("Connection refused by the server..")
			print("Let me sleep for 5 seconds")
			print("ZZzzzz...")
			time.sleep(5)
			print("Was a nice sleep, now let me continue...")
			continue
	return page

""" waste too much space
def cache_fetch_page(db, url):
	print(url) #TEST ONLY
	\"""cache all the fetched page information to avoid fetch it again
	Args:
		db: a dbm database
		url: a url try to get
	Return:
		corresponding page information
	\"""
	try:
		return db[url]
	except KeyError:
		page = get_page(url)
		db[url] = page.content
		return page.content
"""

def get_categories_tags():
	"""get all the fact tag from the webpage https://www.snopes.com/archive/ ONLY
	TODO: add a validation for the accessiability of the webpage
	Args:
		NONE
	Return:
		a list of all kind of fact tags in the website snopes
	"""
	URL = "https://www.snopes.com/archive/"
	page = get_page(URL)
	soup = BeautifulSoup(page.content, 'html.parser')
	for i in soup.find_all('h3', class_="title"):
		if i.get_text().lower() == "fact check archive":
			fact_categories_section = i.parent.next_sibling
			continue

	fact_categories = []
	for t in fact_categories_section.children:
		fact_categories.append(t.get_text())

	return fact_categories

def get_rating_tags():
	"""get all the fact tag from the webpage https://www.snopes.com/archive/ ONLY
	TODO: add a validation for the accessiability of the webpage
	Args:
		NONE
	Return:
		a list of all kind of fact tags in the website snopes
	"""
	URL = "https://www.snopes.com/archive/"
	page = get_page(URL)
	soup = BeautifulSoup(page.content, 'html.parser')
	for i in soup.find_all('h3', class_="title"):
		if i.get_text().lower() == "fact check by rating":
			fact_tag_section = i.parent.next_sibling
			continue

	fact_tags = []
	for t in fact_tag_section.children:
		fact_tags.append(t.get_text())

	return fact_tags


def find_rating(soup):
	"""get the rating information from the given soup
	Args:
		the soup of corresponding webpage
	return:
		the rating information
	"""
	for s in soup.find_all('h3', class_="section-break"):
		if s.get_text().lower() == 'rating':
			return s.next_sibling.next_sibling.get_text().strip().lower()
	return ""

def find_claim(soup):
	"""get the claim information from the given soup
	Args:
		the soup of corresponding webpage
	return:
		the claim information
	"""
	for s in soup.find_all('h3', class_="section-break"):
		if s.get_text().lower() == 'claim':
			return s.next_sibling.next_sibling.get_text().strip()
	return ""

def find_original_link(soup):
	"""get the claim information from the given soup
	Args:
		the soup of corresponding webpage
	return:
		list of original link
	"""
	urls = []
	for s in soup.find_all('h3', class_="section-break"):
		if s.get_text().lower() == 'origin':
			if s.find_next('p'):
				a_tags = s.find_next('p').find_all('a', href = True)
				if s.find_next('p').find_next('p'):
					a_tags += s.find_next('p').find_next('p')\
						.find_all('a', href = True)
				for a in a_tags:
					urls.append(a['href'])
				return urls
	return urls

def get_original_urls(article_soup, is_claim):
	"""get all the original urls on the article and corresponding number of paragraph
	Args:
		beautiful soup of article
	Return:
		[(link, num_of_paragraph)]
	"""
	result = []
	if is_claim:
		article_paragraph = \
		article_soup.find_all('div', class_='article-text-inner')[0].find_all('p')[1:]
	else:
		article_paragraph = \
		article_soup.find_all('div', class_='article-text-inner')[0].find_all('p')
	if is_claim:
		for i in range(len(article_paragraph)):
			p = article_paragraph[i]
			urls = p.find_all('a', href = True)
			for url in urls:
				if not ("https://www.snopes.com/" in url['href']\
				 or "http://www.snopes.com/" in url['href']):
					result.append((url['href'], i+1)) # i + 1 for adjust the index of the paragraph
	return result


def get_articles_info(soup):
	"""get all the links to the article listed in current page
	*** Assuming the link is a correct link ***
	Args:
		beautiful soup of link to the pages list all articles
	Retrun:
		a list of all article links
		a list of all titles of these articles
		a list of corresponding categories of these articles
		a list of corresponding article dates
	"""

	articles_urls = []
	articles_titles = []
	articles_categories = []
	articles_date = []
	articles_claim = []
	articles_rating = []
	articles_origin_url = []

	for t in soup.find_all('a', class_="article-link", href=True):
		if t.find('h2'):
			article_page = get_page(t['href'])
			article_soup = BeautifulSoup(article_page.content, 'html.parser')
			claim = find_claim(article_soup)
			articles_claim.append(claim)
			articles_rating.append(find_rating(article_soup))
			articles_origin_url.append(get_original_urls(article_soup, claim))

			articles_urls.append(t['href'])
			articles_titles.append(t.find('h2').get_text())
			if t.find_all('div', class_="breadcrumbs"):
				articles_categories.append(t.find_all(\
					'div', class_="breadcrumbs")[0].get_text().strip())
			else:
				articles_categories.append("")
			if t.find('span', class_="article-date"):
				articles_date.append(t.find('span', class_="article-date").get_text())
			else:
				articles_date.append("")
	return [articles_urls, articles_titles, articles_categories, \
	articles_date, articles_claim, articles_origin_url, articles_rating]

def get_next_page(soup):
	"""get the url of next page
	*** Assuming the link is a correct link ***
	Args:
		link to the current page
	Return:
		the url link to the next page
	"""
	find_next = soup.find('i', class_="fa fa-chevron-right")
	if find_next:
		return find_next.parent['href']
	else:
		return 0

def write_url_info(articles_info, file):
	"""write the article information on current page into file
	Args:
		articles_info: list of article informations
	Return:
		None
	"""
	EMPTY_LENGTH = 0
	articles_urls, articles_titles, articles_categories, \
	articles_date, articles_claim, articles_origin_url, articles_rating = articles_info
	for i in range(len(articles_urls)):
		if articles_origin_url[i] is not None:
			line = [articles_rating[i]]
			line.append(articles_urls[i])
			line.append(articles_titles[i])
			line.append(basic_clean(articles_categories[i]))
			line.append(articles_date[i])
			line.append(articles_claim[i])
			for j in range(len(articles_origin_url[i])):
				url, index_paragraph = articles_origin_url[i][j]
				write_line = [] + line
				#write_line = 
				write_line.append(url)
				write_line.append(index_paragraph)
				write_line.append(j==0) #check if this is the first link on this webpage
				#line.append(str(len(articles_origin_url[i]) != EMPTY_LENGTH))
				with open(file, 'a', newline='', encoding='utf-8') as f:
					csv_writer = csv.writer(f)
					#line = [l.encode('utf-8') for l in line]
					try:
						csv_writer.writerow(write_line)
					except Exception as e:
						print(e)

def main(output_file):
	#initial result
	SPECIAL_NEWS_CATEGORIES = {'Fraud & Scams':'fraud', 'Fauxtography': 'photos', \
	'Old Wives\' Tales': 'oldwivestales', 'Questionable Quotes': 'quotes', \
	'Risque Business':'nsfw', 'September 11th': 'september-11', 'War/Anti-War': 'war'}

	result_file_name = os.getcwd() + "/" + output_file #snopes_parsed_info_phase1.csv"
	header_names = \
	"fact_rating_phase1,snopes_url_phase1,article_title_phase1,article_category_phase1," +\
	"article_date_phase1,article_claim_phase1,article_origin_url_phase1," +\
	"index_paragraph_phase1,page_is_first_citation_phase1\n"

	# initial a dbm, simple database, in order to cache all the already fetched page
	# don't have to open that page again for geting information
	#db = dbm.open('cache', 'c')

	with open(result_file_name, 'w') as f:
		f.write(header_names)

	basic_url = "https://www.snopes.com/fact-check/rating/"
	fact_tags = get_rating_tags()

	for fact_tag in fact_tags:
		fact_url = basic_url + '-'.join(fact_tag.lower().split(' '))
		while(fact_url):
			page = get_page(fact_url)
			soup = BeautifulSoup(page.content, 'html.parser')
			articles_info = get_articles_info(soup)
			write_url_info(articles_info, result_file_name)
			fact_url = get_next_page(soup)
	#db.close()

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='fetching the article information from snopes')
	parser.add_argument('output_file', type=str, help='the output file')
	args = parser.parse_args()
	main(args.output_file)