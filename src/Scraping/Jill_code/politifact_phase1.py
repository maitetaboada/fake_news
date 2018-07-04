from bs4 import BeautifulSoup
import requests
import os
import re
import time
#import dbm
import csv

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

def get_next_page(soup):
	"""get the url of next page
	*** Assuming the link is a correct link ***
	Args:
		soup to the current page
	Return:
		the suffix link to the next page, the url to next page is "page + find_next['href']"
	"""
	find_next = soup.find('a', class_="step-links__next", href = True)
	if find_next:
		return find_next['href']
	else:
		return 0

def get_all_article_links(soup):
	""" get the article links to the pages on politifacts

	Args:
		soup on the current page
	Return:
		the suffix links to the pages with article source
	"""
	links_suffix = []
	article_links_div = soup.find_all('div', class_="scoretable__item")
	for links_div in article_links_div:
		links_suffix.append(links_div.find('a', class_="link", href=True)['href'])
	return links_suffix

def get_page_info(article_soup):
	"""
	TODO: consider exceptions (such as: no claim, no fact_tag...)
	Args:
		article soup
	Returns:
	[fact_tag, claim, claim_citation, published_date, researched_by, edited_by,\
	categories, original_urls]

	"""
	if article_soup.find('h1', class_='article__title'):
		article_title = article_soup.find('h1', class_='article__title')\
		.get_text().strip()
	else:
		article_title = ""
	if article_soup.find('img', class_='statement-detail', alt=True):
		fact_tag = article_soup.find('img', class_='statement-detail', alt=True)['alt']
	else:
		fact_tag = ""
	if article_soup.find('div', class_='statement__text'):
		claim = article_soup.find('div', class_='statement__text').get_text().strip()
	else:
		claim = ""
	if article_soup.find('p', class_='statement__meta'):
		claim_citation = article_soup.find('p', class_='statement__meta').get_text().strip()
	else:
		claim_citation = ""
	
	# Assuming that the following information will contained on each page in correct order
	about_statement = article_soup.find('div', class_='widget__content').find_all('p')
	published_date = " ".join(about_statement[0].get_text().strip().split(' ')[1:]).strip()
	researched_by = " ".join(about_statement[1].get_text().strip().split(' ')[2:]).strip()
	edited_by = " ".join(about_statement[2].get_text().strip().split(' ')[2:]).strip()
	categories = " ".join(about_statement[3].get_text().strip().split(' ')[1:]).strip()
	original_urls = []
	for s in about_statement[4:]:
		page_is_first_citation = (len(original_urls) == 0) #if the list hasn't been added any item, this is the first link
		if s.find('a', href=True):
			if "http://www.politifact.com/" not in s.find('a', href=True)['href']:
				original_urls.append((s.find('a', href=True)['href'], page_is_first_citation))

	return [fact_tag, article_title, claim, claim_citation, published_date, researched_by, edited_by,\
	categories, original_urls]

def write_url_info(article_info, url, file):
	"""write the information to output file correctly
	
	Args:
		article_info: (List) the needed information of the corresponding page
		url: (String) the url on the politifact.com
		file: (String) the output file
	
	Return:
		None
	"""
	
	original_urls = article_info[-1]
	for l in original_urls:
		line = [url] + article_info[:-1] + list(l)
		with open(file, 'a', newline='', encoding='utf-8') as f:
			csv_writer = csv.writer(f)
			#line = [l.encode('utf-8') for l in line]
			try:
				csv_writer.writerow(line)
			except Exception as e:
				print(e)

def main(output_file):
	result_file_name = os.getcwd() + "/" + output_file #snopes_parsed_info_phase1.csv"
	header_names = \
	"politifact_url_phase1,fact_tag_phase1,article_title_phase1,article_claim_phase1," +\
	"article_claim_citation_phase1,article_published_date_phase1,article_researched_by_phase1," +\
	"article_edited_by_phase1,article_categories_phase1,original_url_phase1,page_is_first_citation_phase1\n"

	# initial a dbm, simple database, in order to cache all the already fetched page
	# don't have to open that page again for geting information
	#db = dbm.open('cache', 'c')

	with open(result_file_name, 'w') as f:
		f.write(header_names)

	WEBSITE = "http://www.politifact.com/truth-o-meter/statements/"
	POLITIFACT_ROOT = "http://www.politifact.com"
	politifact_url = WEBSITE
	while politifact_url:
		page = get_page(politifact_url)
		soup = BeautifulSoup(page.content.decode('utf-8'), 'html.parser')
		if get_next_page(soup):
			politifact_url = WEBSITE + get_next_page(soup)
		else:
			politifact_url = ""
		all_article_links_suffix = get_all_article_links(soup)
		for suffix in all_article_links_suffix:
			link = POLITIFACT_ROOT + suffix
			article_page = get_page(link)
			article_soup = BeautifulSoup(article_page.content.decode('utf-8'), 'html.parser')
			page_info = get_page_info(article_soup)
			write_url_info(page_info, link, output_file)

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='fetching the article information from politifact')
	parser.add_argument('output_file', type=str, help='the output file')
	args = parser.parse_args()
	main(args.output_file)


