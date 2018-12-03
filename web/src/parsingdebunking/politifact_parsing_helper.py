from bs4 import BeautifulSoup
import requests
import os
import re
import time
import csv
import datetime

def get_page(url):
	print(url)
	page = ''
	count = 0
	while page == '' and count < 100:
		count += 1
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
		the suffix link to the next page, the url to next page is 
		"page + find_next['href']"
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

def convert_timestamp(timestamp):
	"""convert the original timestamp format on polifacts to an acceptable one
	Args:
		[STR] timestamp: for example: 'Monday, July 2nd, 2018 at 4:18 p.m.'
	Return:
		[STR] formated timestamp: for example: "2 July 2018"
	"""
	print(timestamp)
	formated_timestamp = re.sub(" at.*", "", timestamp)
	formated_timestamp = formated_timestamp.strip()
	month = formated_timestamp.split(',')[1].split()[0]
	day = formated_timestamp.split(',')[1].split()[1]
	day = re.sub('[a-z].*', "", day.lower())
	year = formated_timestamp.split(',')[-1].strip()
	return " ".join([day, month, year])

def get_page_info(url, time_label):
	"""
	TODO: consider exceptions (such as: no claim, no fact_tag...)
	Args:
		article soup
	Returns:
	[url, fact_tag, claim, claim_citation, published_date, researched_by, edited_by,\
	categories, original_urls]

	"""
	to_continue = True
	article_page = get_page(url)
	article_soup = BeautifulSoup(article_page.content.decode('utf-8'), 'html.parser')
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
	if time_label:
		if (datetime.datetime.strptime(convert_timestamp(published_date), "%d %B %Y")\
		 < datetime.datetime.strptime(time_label, "%Y-%m-%d")):
			to_continue = False

	researched_by = " ".join(about_statement[1].get_text().strip().split(' ')[2:]).strip()
	edited_by = " ".join(about_statement[2].get_text().strip().split(' ')[2:]).strip()
	categories = " ".join(about_statement[3].get_text().strip().split(' ')[1:]).strip()
	original_urls = []
	for s in about_statement[4:]:
		page_is_first_citation = (len(original_urls) == 0) #if the list hasn't been added any item, this is the first link
		if s.find('a', href=True):
			if "http://www.politifact.com/" not in s.find('a', href=True)['href']:
				original_urls.append((s.find('a', href=True)['href'], page_is_first_citation))
	#print("the title is: ")
	#print(article_title)
	return [url, fact_tag, article_title, claim, claim_citation, published_date, researched_by, edited_by,\
	categories, original_urls], to_continue

def write_url_info(article_info, file_path, is_first_row, to_parse_origin, is_relative=True):
	"""write the information to output file correctly
	
	Args:
		article_info: (List) the needed information of the corresponding page
		url: (String) the url on the politifact.com
		file: (String) the output file
	
	Return:
		None
	"""
	if is_first_row:
		header_names = \
		"politifact_url_phase1,fact_tag_phase1,article_categories_phase1,article_claim_citation_phase1," +\
		"article_published_date_phase1,article_researched_by_phase1,article_edited_by_phase1," +\
		"citation_order_phase1,claim_title_phase1,claim_phase1,original_url_phase1,page_is_first_citation_phase1\n"

		if to_parse_origin:
			header_names = header_names.rstrip() + ',error_phase2,original_article_text_phase2,' +\
			 'article_title_phase2,publish_date_phase2,author_phase2\n'

		with open(file_path, 'w') as f:
			f.write(header_names)

	original_urls = article_info[-1]
	for l in original_urls:
		url, _ = l
		line = article_info[:-1] + [url]
		if to_parse_origin:
			if is_relative:
				from .parsing_original_web_helper import get_origin_article_info
			else:
				from parsing_original_web_helper import get_origin_article_info
			original_article_info = get_origin_article_info(url)
			line += original_article_info
		with open(file_path, 'a', newline='', encoding='utf-8') as f:
			csv_writer = csv.writer(f)
			try:
				csv_writer.writerow(line)
			except Exception as e:
				print(e)

def parsing_whole_wepages(time_label, output_path):
	import datetime
	now = datetime.datetime.now()
	timestamp = "_".join([str(x) for x in [now.year, now.month, now.day]])
	
	output_file1 = "politifact_phase1_raw_" + timestamp + ".csv"
	output_file2 = "politifact_phase2_raw_" + timestamp + ".csv"
	output_filename1 = output_path + "/" + output_file1 #put the result into a relative path
	output_filename2 = output_path + "/" + output_file2 #put the result into a relative path

	WEBSITE = "http://www.politifact.com/truth-o-meter/statements/"
	POLITIFACT_ROOT = "http://www.politifact.com"
	politifact_url = WEBSITE
	is_first_row = True
	to_continue = True
	while politifact_url and to_continue:
		page = get_page(politifact_url)
		soup = BeautifulSoup(page.content.decode('utf-8'), 'html.parser')
		if get_next_page(soup):
			politifact_url = WEBSITE + get_next_page(soup)
		else:
			politifact_url = ""
		all_article_links_suffix = get_all_article_links(soup)
		for suffix in all_article_links_suffix:
			link = POLITIFACT_ROOT + suffix
			#article_page = get_page(link)
			#article_soup = BeautifulSoup(article_page.content.decode('utf-8'), 'html.parser')
			page_info, to_continue = get_page_info(link, time_label)
			if to_continue:
				write_url_info(page_info, output_filename1, is_first_row, False, False)
				write_url_info(page_info, output_filename2, is_first_row, True, False)
				is_first_row = False
			else:
				break
	return output_filename1, output_filename2
"""
if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='fetching the article information from politifact')
	parser.add_argument('output_file', type=str, help='the output file')
	args = parser.parse_args()
	main(args.output_file)
"""

