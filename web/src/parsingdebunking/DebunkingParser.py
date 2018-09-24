class DebunkingParser(object):
	def __init__(self, debunking_web_name):
		self.debunking_web_name = debunking_web_name

	def parsing_web(self, time_label, url, to_parse_origin, output_dir):
		import re
		import tldextract
		import csv
		file_name = ""
		if self.debunking_web_name == "snopes" \
		and tldextract.extract(url).domain == "snopes":
			from .snopes_parsing_helper import get_page_info, write_url_info
			file_name = "_".join(re.sub(r'.*snopes.com/', "", url).strip('/').split("/"))\
			 + "_" + time_label + ".csv"
			output_file_name = output_dir +  file_name

		elif self.debunking_web_name == "politifact" \
		and tldextract.extract(url).domain == "politifact":
			from .politifact_parsing_helper import get_page_info, write_url_info
			file_name = "_".join(re.sub(r'.*politifact.com/', "", url).strip('/').split("/"))\
			 + "_" + time_label + ".csv"
			output_file_name = output_dir +  file_name

		elif self.debunking_web_name == "emergent" \
		and tldextract.extract(url).domain == "emergent":
			from .emergent_parsing_helper import get_page_info, write_url_info
			file_name = "_".join(re.sub(r'.*emergent.info/', "", url).strip('/').split("/"))\
			+ "_" + time_label + ".csv"
			output_file_name = output_dir +  file_name

		article_info = get_page_info(url)
		write_url_info(article_info, output_file_name, True, to_parse_origin)

		return file_name
