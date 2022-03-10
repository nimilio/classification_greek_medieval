import os
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

dataset = "./ep/test/"
documents = [file for file in os.listdir(dataset) if file.endswith('.txt')]
nlp = spacy.load("el_core_news_sm")
for document in documents:
	with open(dataset+document,"r",encoding='utf-8') as f:
		content = f.read()
	tok = word_tokenize(content)
	for i,e in enumerate (tok):
		if "αι" in e:
			tok[i] = e.replace("αι", "ᾳ")
		elif "ηι" in e:
			tok[i] = e.replace("ηι", "ῃ")
		elif "ωι" in e:
			tok[i] = e.replace("ωι", 'ῳ')
		elif "ῶι" in e:
			tok[i] = e.replace("ῶι", 'ῷ')
		elif "ῆι" in e:
			tok[i] = e.replace("ῆι", 'ῇ')
		elif "ᾶι" in e:
			tok[i] = e.replace("ῆι", 'ᾷ')
	joined = TreebankWordDetokenizer().detokenize(tok)
	with open("./ep_tok/"+document, 'w') as file:
		file.write(joined)








