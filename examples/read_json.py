import json
import bz2
import nltk.data
import re
import os
import sys
import string

f=open('Reddit_data_sentences.txt','w')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

for file in os.listdir(sys.argv[1]):
	filepath=sys.argv[1]+file
	contents=bz2.BZ2File(filepath, "r")
	for sent_read in contents:
		data = json.loads(sent_read)
		try:
			sent_content=str(data['body'])
			sent_content=sent_content.strip()
			if sent_content=='[deleted]':
				continue
			#print sent_content,"\n"
			sentence_list=tokenizer.tokenize(sent_content)
			for sentence in sentence_list:
				if sentence.isspace():
					continue
				sentence=re.sub(r'[(){}\[\]\'\"\r\n]','',sentence)
				#for ch in ['(',')','[',']','{','}','\'','\"','\n','\r']:
				#	sentence=sentence.replace(ch,'')
				sentence=(sentence.lower()).rstrip('?:!.,;')
				if sentence.isspace():
					continue
				f.write(sentence)
				f.write('\n')
				#print line
		except (UnicodeDecodeError, UnicodeEncodeError) as e:
			continue
f.close()

