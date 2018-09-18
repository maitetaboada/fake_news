# Overall Explaination
This parsing debunking file is used to support the "SCRAPE FACT-CHECKING WHENEBSITES" section.
There are two main functions it achieved:
- It supports the user to parse only one webpage and clean those parsed data.
- It supports the server to update the file time by time, and parsed the file from now to last time we parsed the file.

# Prerequirement:
- For parsing one webpage, it has to combined with the UI provided in the html file
	- We also required that there is a folder named as "temporary_files" in the path "path/to/Flask/Flask_UI/", which means we required a path as **path/to/Flask/Flask_UI/temporary_files**
- For parsing all three debunking wepages, we required:
	- There is a file named as "large_files" in the path "path/to/Flask/Flask_UI/", which means we required a path as **path/to/Flask/Flask_UI/large_files**
	- Some old files named as the format "webname_phase#_raw(or clean)_year_month_day.csv", e.g.: "snopes_phase1_raw_2018_7_13.csv".
	**The easiest way to get the "large_files.zip" is from Vault "Discourse-Lab/Data/Fake_news/Fatemeh/Jill/web_demo_files"**
- For recording the users' parsing requirements, we need a folder named as "logs" to store the log files. Thus, we also need a directory as **path/to/Flask/Flask_UI/logs**
- We suppose the server that running the webpage is a **"Linux"** System.

# Individual file explaination
- clear.sh:
	- **Note**:
		Before you use it, you should open the file, and change the path to **ABSOLUTE PATH TO "temporary_files** folder, which is "/home/yzhou/web/Flask/Flask_UI/temporary_files/" now. please change it!
	- It used to clean the file every **hour**. For setting it, you should set it in crontab:
		Steps: 
		1. "crontab -e"
		2. add "0 * * * * bash absolute_PATH/to/clear.sh" at the very bottom
	- Of course, you can run it manually by using "bash PATH/to/clear.sh", which will clean the "temporary_files" folder.
- parsingData.sh
	- **Note**:
		Before you use it, you should open the file, and change the path to **ABSOLUTE PATH TO "parsing_whole_fact_checker.py** folder, which is "/home/yzhou/web/src/parsingdebunking/parsing_whole_fact_checker.py" now. please change it!
	- Used to update the existing entire archive files every **Sunday**. For setting it, you should set it in crontab:
		Steps: 
		1. "crontab -e"
		2. add "0 0 * * 0 bash absolutePATH/to/clear.sh" at the very bottom
	- Of course, you can run it manually by using "bash PATH/to/parsingData.sh", which will parse all the debunking webpages from now to the time_label on the file names in the folder large_files.
- data_cleaner:
	- Clean strange texts and failed parsing webpages
- DebunkingParser.py
	- For paring the individual debunking webpage
- emergent_parsing_helper.py
	- Help in parsing the www.emergent.info
- snopes_parsing_helper.py
	- Help in parsing the www.snopes.com
- politifact_parsing_helper.py
	- Help in parsing the www.politifact.com
- parsing_original_web_helper.py
	- Help in parsing with the original URL
- parsing_whole_fact_checker.py
	- Help in parsing the whole fact webpages time by time




