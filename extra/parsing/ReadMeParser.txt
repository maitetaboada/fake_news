You might have to install stanford from nltk.parse and "Tree" from nltk.

Link to download the parser:
https://nlp.stanford.edu/software/stanford-parser-full-2017-06-09.zip

Put the following files in a folder and name it "jars":
stanford-parser-3.8.0-models.jar
stanford-parser.jar
META-INF

Run this command from the stanford-parser-full-2017-06-09 folder:
unzip stanford-parser-3.8.0-models.jar

Copy the "edu" folder which is made in result of the command, to the jars folder

Run the parser.py script and given the input and output file name. The text that needs to be parsed should be present in a column named "text" of the input csv file.

parser.py:
Runs the Stanford parser on the input file.

extract_rules.py:
Extracts the lexicalized and non lexicalized rules from the extracted rules (output of parser.py)

This is how the parser builds the tree:
http://nlp.stanford.edu:8080/parser/



