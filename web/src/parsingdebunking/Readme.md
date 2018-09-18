# Overall Explaination

This parsing debunking file is used to support the "SCRAPE FACT-CHECKING WHENEBSITES" section.

# Individual file explaination
- clear.sh:
	- **Note**:
		Before you use it, you should open the file, and change the path to **ABSOLUTE PATH TO "temporary_file** folder, which is "/home/yzhou/web/Flask/Flask_UI/temporary_files/" now. please change it!
	- Used to clean the file every **hour**. For setting it, you should set it in crontab:
		Steps: 
		1. "crontab -e"
		2. add "0 * * * * bash absolutePATH/to/clear.sh" at the very bottom
- parsingData.sh
	- **Note**:
		Before you use it, you should open the file, and change the path to **ABSOLUTE PATH TO "parsing_whole_fact_checker.py** folder, which is "/home/yzhou/web/src/parsingdebunking/parsing_whole_fact_checker.py" now. please change it!
	- Used to clean the file every **Sunday**. For setting it, you should set it in crontab:
		Steps: 
		1. "crontab -e"
		2. add "0 0 * * 0 bash absolutePATH/to/clear.sh" at the very bottom
- data_cleaner:
	- clean strange texts and failed parsing webpages
- DebunkingParser.py
	- for paring the individual debunking webpage
- emergent_parsing_helper.py
	- help in parsing the www.emergent.info
- snopes_parsing_helper.py
	- help in parsing the www.snopes.com
- politifact_parsing_helper.py
	- help in parsing the www.politifact.com
- parsing_original_web_helper.py
	- help in parsing with the original URL
- parsing_whole_fact_checker.py
	- help in parsing the whole fact webpages time by time




