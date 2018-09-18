#Overall Explaination

This parsing debunking file is used to support the "SCRAPE FACT-CHECKING WHENEBSITES" section.

#Individual file explaination
- clear.sh:
	- *Note*:
		Before you use it, you should open the file, and change the path to *ABSOLUTE PATH TO "temporary_file* folder, which is "/home/yzhou/web/Flask/Flask_UI/temporary_files/\*" now. please change it!
	- Used to clean the file every hour. For setting it, you should set it in crontab:
		Steps: 
		1. "crontab -e"
		2. add "59 * * * * bash absolutePATH/to/clear.sh"

