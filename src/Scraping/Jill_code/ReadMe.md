#### Environment
1. Python3.6 (https://www.python.org/downloads/)
2. python package:
	- time	(Default installed package)
	- os	(Default installed package)
	- re	(pip3 install re)
	- bs4	(pip3 install bs4)
	- csv	(pip3 install csv)
	- newspaper	(pip3 install newspaper3k)
3. For parsing from emergent.info:
	- dryscrape	(pip3 install dryscrape)

#### Usage
1. For all the file with suffix "Phase1", we tried to parse the information from different webpage.
	- python3 snopes_phase1.py output_file_name
	- python3 emergent_phase1.py output_file_name
	- python3 politifact_phase1.py output_file_name


2. For all the file with suffix "Phase2", we tried to parse the original article from different webpages parsed from phase1.
	- python3 snopes_phase2.py input_file_name.csv output_file_name.csv
	- python3 emergent_phase2.py input_file_name.csv output_file_name.csv
	- python3 politifact_phase2.py input_file_name.csv output_file_name.csv

3. For all the parsed original article, we can clean it by using data_cleaner.py.
	- python3 data_cleaner.py input_file_name.csv output_file_name.py website_name
(Notes: the website names are the domain name of websites. For example, for snopes.com, the website_name should be "snopes")
Notes: now the code only support the snopes.com, politifact.com, and emergent.info, which is a blog.
