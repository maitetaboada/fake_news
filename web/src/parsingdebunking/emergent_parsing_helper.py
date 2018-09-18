import dryscrape
from bs4 import BeautifulSoup
import os
import re
import time
import csv
import datetime

def get_soup(url):
	print(url)
	soup = ''
	count = 0
	while soup == '' and count < 100:
		count += 1
		try:
			time.sleep(1)
			session = dryscrape.Session()
			session.visit(url)
			response = session.body()
			soup = BeautifulSoup(response, 'html.parser')
		except:
			print("Connection refused by the server..")
			print("Let me sleep for 5 seconds")
			print("ZZzzzz...")
			time.sleep(5)
			print("Was a nice sleep, now let me continue...")
			continue
	return soup

def get_categories_suffix(soup):
	""" get all categories suffix link on emergent webpage
	Args:
		soup on emergent root webpage, www.mergent.info
	Return:
		all categories on this webpage
	"""
	categories_suffix = []
	if soup.find_all('a', class_="navigation-link", href=True):
		for i in soup.find_all('a', class_="navigation-link", href=True):
			if 'category' in i['href'].split('/'):
				categories_suffix.append(i['href'])
	return categories_suffix

def get_next_page_suffix(article_link):
	"""get the url of next page
	*** Assuming the link is a correct link ***
	Args:
		soup to the current page
	Return:
		the suffix link to the next page, the url to next page is "page + find_next['href']"
	"""
	article_soup = get_soup(article_link)
	find_next = article_soup.find('div', class_="next-link-holder")
	if find_next:
		return find_next.find('a', href = True)['href']
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
	article_links = soup.find_all('article', class_ ="article article-preview with-truthiness")
	for article_link in article_links:
		links_suffix.append(article_link.find('h2').find('a', href = True)['href'])
	return links_suffix

def get_page_info(article_link):
	"""
	Args:
		article soup
	Returns:
	[article_link, fact_tag, category, claim_description,\
	 article_tags, article_date,  claim, article_tracking_body, original_url]

	"""
	article_soup = get_soup(article_link)
	if article_soup.find('span', class_="truthiness-value"):
		fact_tag = article_soup.find('span', class_="truthiness-value")\
		.get_text().strip()
	else:
		fact_tag = ""
	if article_soup.find('h1'):
		claim = article_soup.find('h1').get_text()
	else:
		claim = ""
	if article_soup.find('p', class_="article-content"):
		claim_description = article_soup.find('p', class_="article-content").get_text().strip()
	else:
		claim_description = ""
	if article_soup.find('div', class_="article-tags"):
		# get all article tags
		article_tags = article_soup.find('div', class_="article-tags")
		article_tags = [t.get_text() for t in article_tags.find_all('a')]
		article_tags = "&".join(article_tags)
	else:
		article_tags = ""

	original_url = ""
	article_date = ""
	article_tracking_body = ""
	claim_links = article_soup.find_all('p', class_="tracking")
	if claim_links:
		for l in claim_links:
			if "Originating Source" in l.get_text():
				if l.find('a',href=True):
					original_url = l.find('a',href=True)['href']
				if l.find('span'):
					if l.find('span').find_all('span'):
						article_date = l.find('span').find_all('span')[-1].get_text()
				if article_soup.find('span', class_="tracking-body"):
					article_tracking_body = l.find('span', class_="tracking-body").get_text()

	""" not be able to parse total shares value
	total_share = ""

	if article_soup.find_all('span', class_="shares-value"):
		print(article_soup.find_all('span', class_="shares-value"))
		total_share = article_soup.find_all('span', class_="shares-value")[-1].get_text()
	"""
	category = article_soup.find('h2', class_="page-title")
	if not category:
		category = ""
	else:
		category = category.get_text()
	return [article_link, category, fact_tag, claim, claim_description,\
	 article_tags, article_date, article_tracking_body, original_url]
	

def write_url_info(article_info, file_path, is_first_row, to_parse_origin):
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
		"emergent_url_phase1,claim_category_phase1,fact_tag_phase1,claim_title_phase1," +\
		"claim_description_phase1,claim_tags_phase1,article_date_phase1,"+\
		"claim_body_phase1,original_url_phase1\n"
		if to_parse_origin:
			header_names = header_names.rstrip() + ',error_phase2,original_article_text_phase2,' +\
			 'article_title_phase2,publish_date_phase2,author_phase2\n'
		with open(file_path, 'w') as f:
			f.write(header_names)

	line = article_info
	if to_parse_origin:
		from .parsing_original_web_helper import get_origin_article_info
		original_article_info = get_origin_article_info(line[-1])
		line += original_article_info
	with open(file_path, 'a', newline='', encoding='utf-8') as f:
		csv_writer = csv.writer(f)
		#line = [l.encode('utf-8') for l in line]
		try:
			csv_writer.writerow(line)
		except Exception as e:
			print(e)

def parsing_whole_wepages(output_path):
	import datetime
	from parsing_original_web_helper import get_origin_article_info
	

	now = datetime.datetime.now()
	timestamp = "_".join([str(x) for x in [now.year, now.month, now.day]])
	
	output_file1 = "emergent_phase1_raw_" + timestamp + ".csv"
	output_file2 = "emergent_phase2_raw_" + timestamp + ".csv"
	result_file1_name = output_path + "/" + output_file1 #put the result into a relative path
	result_file2_name = output_path + "/" + output_file2 #put the result into a relative path

	# initial a dbm, simple database, in order to cache all the already fetched page
	# don't have to open that page again for geting information
	#db = dbm.open('cache', 'c')

	header_names1 = \
	"emergent_url_phase1,fact_tag_phase1,claim_category_phase1,claim_description_phase1," +\
	"claim_tags_phase1,article_date_phase1,claim_title_phase1,"+\
	"claim_body_phase1,original_url_phase1\n"

	header_names2 = \
	header_names1.rstrip() + ',error_phase2,original_article_text_phase2,' +\
			 'article_title_phase2,publish_date_phase2,author_phase2\n'

	with open(result_file1_name, 'w') as f:
		f.write(header_names1)

	with open(result_file2_name, 'w') as f:
		f.write(header_names2)

	EMERGENT_ROOT = "http://www.emergent.info"
	# get all categories
	root_soup = get_soup(EMERGENT_ROOT)
	categories_suffix = get_categories_suffix(root_soup)
	for c in categories_suffix:
		emergent_cat_url = EMERGENT_ROOT + c
		# find all suffix links on current pages at first
		emergent_cat_soup = get_soup(emergent_cat_url)
		article_suffix = get_all_article_links(emergent_cat_soup)
		# parsing over all links on current pages
		for suffix in article_suffix:
			article_link = EMERGENT_ROOT + suffix
			article_link, category, fact_tag, claim, claim_description,\
			article_tags, article_date, article_tracking_body, original_url = \
			get_page_info(article_link)
			line = [article_link, fact_tag, category, claim_description, article_tags, \
			article_date, claim, article_tracking_body, original_url]
			with open(result_file1_name, 'a', newline='', encoding='utf-8') as f:
				csv_writer = csv.writer(f)
				#line = [l.encode('utf-8') for l in line]
				try:
					csv_writer.writerow(line)
				except Exception as e:
					print(e)
			
			original_article_info = get_origin_article_info(line[-1])
			line += original_article_info
			with open(result_file2_name, 'a', newline='', encoding='utf-8') as f:
				csv_writer = csv.writer(f)
				#line = [l.encode('utf-8') for l in line]
				try:
					csv_writer.writerow(line)
				except Exception as e:
					print(e)
							

		# after parsing over from all links, there might be more links, which can
		# be tracked by next page
		while(get_next_page_suffix(article_link)):
			article_link = EMERGENT_ROOT + get_next_page_suffix(article_soup)
			article_link, category, fact_tag, claim, claim_description,\
			article_tags, article_date, article_tracking_body, original_url = \
			get_page_info(article_link)
			line = [url, fact_tag, category, claim_description, article_tags, \
			article_date, claim, article_tracking_body, original_url]
			#phase1
			with open(result_file1_name, 'a', newline='', encoding='utf-8') as f:
				csv_writer = csv.writer(f)
				#line = [l.encode('utf-8') for l in line]
				try:
					csv_writer.writerow(line)
				except Exception as e:
					print(e)

			#phase2
			original_article_info = get_origin_article_info(line[-1])
			line += original_article_info
			with open(result_file2_name, 'a', newline='', encoding='utf-8') as f:
				csv_writer = csv.writer(f)
				#line = [l.encode('utf-8') for l in line]
				try:
					csv_writer.writerow(line)
				except Exception as e:
					print(e)


	return result_file1_name, result_file2_name



