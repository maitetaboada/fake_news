"""
This file is to parse the whole fact checker webpages
"""

import os
import glob
import re
from data_cleaner import data_clean
from zipfile import ZipFile

def _merge_files(old_file, new_file):
	print("merging " + old_file + " into " + new_file)
	with open(old_file) as o:
		next(o)
		for line in o:
			with open(new_file, 'a', newline='', encoding='utf-8') as f:
				f.write(line)

def _count_file_length(file):
	count = 0
	try:
		with open(file, "r") as f:
			for l in f:
				count += 1
	except Exception as e:
		print(e)
	return count

def main(webname):
	input_path = "../../Flask/Flask_UI/large_files"
	partial_file_name1 = webname + "_phase1_raw*.csv"
	partial_file_name2 = webname + "_phase2_raw*.csv"
	old_phase1_file1 = glob.glob(input_path + "/" + partial_file_name1)[0]
	old_phase1_file2 = glob.glob(input_path + "/" + partial_file_name2)[0]

	partial_cleanfile_name1 = webname + "_phase1_clean*.csv"
	partial_cleanfile_name2 = webname + "_phase2_clean*.csv"
	old_phase1_cleanfile1 = glob.glob(input_path + "/" + partial_cleanfile_name1)[0]
	old_phase1_cleanfile2 = glob.glob(input_path + "/" + partial_cleanfile_name2)[0]
	time_label = "-".join(old_phase1_file1.strip(".csv").split("_")[-3:])
	#time_label = "2018-9-17" #[Test Purpose]
	# phase1
	print("Time label of the old files:" + str(time_label))
	if webname == "snopes":
		from snopes_parsing_helper import parsing_whole_wepages
		print("Parsing Snopes")
		new_phase1_file, new_phase2_file = parsing_whole_wepages(time_label, input_path)

	elif webname == "politifact":
		from politifact_parsing_helper import parsing_whole_wepages
		print("Parsing Politifact")
		new_phase1_file, new_phase2_file = parsing_whole_wepages(time_label, input_path)

	elif webname == "emergent":
		from emergent_parsing_helper import parsing_whole_wepages
		print("Parsing Emergent")
		new_phase1_file, new_phase2_file = parsing_whole_wepages(input_path)
	
	print("Preparing file names")
	output_file_name1 = new_phase1_file.split("/")[-1]
	output_file_name1 = re.sub('raw', "clean", output_file_name1)
	output_cleanfile_name1 = data_clean(new_phase1_file, input_path, webname, False, output_file_name1) #phase1_clean
	output_file_name2 = new_phase2_file.split("/")[-1]
	output_file_name2 = re.sub('raw', "clean", output_file_name2)
	output_cleanfile_name2 = data_clean(new_phase2_file, input_path, webname, True, output_file_name2) #phase1_clean
	print("merging the old files now")
	if webname != "emergent":
		_merge_files(old_phase1_file1, new_phase1_file)
		_merge_files(old_phase1_file2, new_phase2_file)
		_merge_files(old_phase1_cleanfile1, output_cleanfile_name1)
		_merge_files(old_phase1_cleanfile2, output_cleanfile_name2)
	
	print("making the zip files")
	#making zip file
	for f in [new_phase1_file, new_phase2_file, output_cleanfile_name1, output_cleanfile_name2]:
		try:
			z = re.sub(".csv", ".zip", f)
			with ZipFile(z,'w') as zip: 
			# writing each file one by one 
				zip.write(f)
		except Exception as e:
			print(e)

	#remove old files:

	if webname == "emergent":
		# if fail to parse complete new emergent webpages
    		if _count_file_length(old_phase1_file1) > _count_file_length(new_phase1_file):
    			old_phase1_file1 = new_phase1_file
    			old_phase1_file2 = new_phase2_file
    			old_phase1_cleanfile1 = output_cleanfile_name1
    			old_phase1_cleanfile2 = output_cleanfile_name2

	print("removing the old files")
	for f in [old_phase1_file1, old_phase1_file2, old_phase1_cleanfile1, old_phase1_cleanfile2]:
		try:
			os.remove(f)
			z = re.sub(".csv", ".zip", f)
			os.remove(z)
		except Exception as e:
			print(e)



if __name__ == "__main__":
	webnames = ["snopes", "politifact", "emergent"]
	#webnames = ["emergent"]
	for n in webnames:
		main(n)
