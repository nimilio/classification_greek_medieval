import os
import pandas as pd
from nltk.tokenize import sent_tokenize
import re

frame_text = []
frame_category = []

frame_allos = []
frame_autos = []
frame_dem = []
frame_indef = []
frame_pers = []
frame_refl = []

frame_coord = []
frame_men = []
frame_particles = []

frame_circum = []
frame_cond = []
frame_purp = []
frame_man1 = []
frame_rel = []
frame_temp_caus = []
frame_inf = []
frame_mean_rel = []

frame_question = []
frame_super = []
frame_excl = []
frame_man2 = []
frame_length = []



path_epigram = "/Users/nikol/Desktop/svm_texts/epigram/test/"
#path_epigram = "/Users/nikol/Desktop/clustering/epigram/"
epigrams = [txt for txt in os.listdir(path_epigram) if txt.endswith('.txt')]

path_hymns = "/Users/nikol/Desktop/svm_texts/hymns/Test/"
#path_hymns = "/Users/nikol/Desktop/clustering/hymn/"
hymns = [txt for txt in os.listdir(path_hymns) if txt.endswith('.txt')]
#h_p = [txt for txt in os.listdir(path_hymns_poetry) if txt.endswith('.txt')]

path_poetry = "/Users/nikol/Desktop/svm_texts/poems/Test/"
#path_poetry = "/Users/nikol/Desktop/clustering/poetry/"
poetry = [txt for txt in os.listdir(path_poetry) if txt.endswith('.txt')]

path_history = "/Users/nikol/Desktop/svm_texts/historical/Test/"
#path_history = "/Users/nikol/Desktop/clustering/historical/"
historical = [txt for txt in os.listdir(path_history) if txt.endswith('.txt')]

path_religion = "/Users/nikol/Desktop/svm_texts/religious/Test/"
#path_religion = "/Users/nikol/Desktop/clustering/religious/"
religious = [txt for txt in os.listdir(path_religion) if txt.endswith('.txt')]

#path_political = "/Users/nikol/Desktop/clustering/political/"
#political = [txt for txt in os.listdir(path_political) if txt.endswith('.txt')]

#path_literature = "/Users/nikol/Desktop/clustering/literature/"
#literature = [txt for txt in os.listdir(path_literature) if txt.endswith('.txt')]

def features(path, category,label,question=';'):

	allos_other = ['ἄλλος','ἄλλου','ἄλλῳ','ἄλλον','ἄλλη','ἄλλης','ἄλλῃ','ἄλλην','ἄλλο','ἄλλοι','ἄλλων','ἄλλοις','ἄλλους','ἄλλαι','ἄλλαις','ἄλλας','ἄλλα']
	autos_self = ['αὐτός','αὐτοῦ','αὐτῷ','αὐτόν','αὐτή','αὐτῆς','αὐτῇ','αὐτήν','αὐτό','αὐτοί','αὐτῶν','αὐτοῖς','αὐτούς','αὐταί','αὐταῖς','αὐτάς','αὐτά']
	demonostrative_pronouns = ['οὗτος','τούτου','τούτῳ','τοῦτον','αὕτη','ταύτης','ταύτῃ','ταύτην','τοῦτο','οὗτοι','τούτων','τούτοις','τούτους','αὗται','ταύταις','ταύτας','ταῦτα','ὅδε','τοῦδε','τῷδε','τόνδε','ἥδε','τῆσδε','τῇδε','τήνδε','τόδε','οἵδε','τῶνδε','τοῖσδε','τούσδε','αἵδε','ταῖσδε','τάσδε','τάδε','ἐκεῖνος','ἐκείνου','ἐκείνῳ','ἐκεῖνον','ἐκείνη','ἐκείνης','ἐκείνῃ','ἐκεῖνην','ἐκεῖνο','ἐκεῖνοι','ἐκείνων','ἐκείνοις','ἐκείνους','ἐκεῖναι','ἐκείναις','ἐκείνας','ἐκεῖνα']
	indefinite_pronouns = ['τινὸς','του','τινὶ','τῳ','τινὰ','τινὲς','τινῶν','τισὶν','τισὶ','τινὰς','τινὰ','ἄττα']
	personal_pronouns = ['ἐγὼ','ἐμοῦ', 'μου','ἐμοί','μοι','ἐμέ', 'με','σὺ','σοῦ', 'σου','σοί','σοι','σέ', 'σε','ἡμεῖς','ἡμῶν','ἡμῖν','ἡμᾶς','ὑμεῖς','ὑμῶν','ὑμῖν','ὑμᾶς']
	reflexive_pronouns = ['ἐμαυτοῦ','ἐμαυτῷ','ἐμαυτὸν','ἡμῶν αὐτῶν','ἡμῖν αὐτοῖς','ἡμᾶς αὐτοὺς']
	coordinating_conjuctions = ['τε', 'τ΄','καί', 'καὶ', 'ἀλλά', 'ἀλλὰ', 'καίτοι',  'οὐδέ', 'οὐδὲ', 'οὐδ΄','μηδέ', 'μηδὲ', 'μηδ΄','οὔτε', 'οὔτ΄','μήτε', 'μήτ΄', 'ἤ', 'ἢ']
	men_indeed = ['μέν', 'μὲν']
	particles = ['ἄν', 'ἂν','ἆρα','γέ','γ΄', 'δ΄', 'δέ', 'δὲ', 'δή', 'δὴ', 'ἕως', 'κ΄', 'κε', 'κέ', 'κὲ', 'κέν', 'κὲν', 'κεν','μά','μέντοι', 'μὴν', 'μήν', 'μῶν', 'νύ', 'νὺ', 'νυ', 'οὖν', 'περ', 'πω', 'τοι']
	circumstantial_markers = ['ἔπειτα', 'ἔπειτ΄','ὅμως', 'ὁμῶς','καίπερ','ἅτε','ἅτ΄']
	conditional_markers = ['εἰ', 'εἴ', 'εἲ', 'ἐάν','ἐὰν']
	purpose = ['ἵνα','ἵν΄']
	manner1 = ['ὅπως']
	relative_clause = ['ὃς','ὅς','οὗ','ᾧ','ὃν','ὅν','oἳ','οἵ','ὧν','οἷς','οὓς','οὕς','ἣ','ἥ','ἧς','ᾗ','ἣν','ἥν','αἵ','αἷς','ἃς','ἅς','ὃ','ὅ','ἃ','ἅ']
	temporal_causal_markers = ['μέκρι', 'ἕως','πρίν','ἐπεί','ἐπειδή','ἐπειδάν','ὅτε','ὅταν']
	inference = ['ὥστε']
	superl = ['τατος', 'τάτου', 'τάτῳ', 'τατον', 'τατοι', 'τάτων', 'τάτοις', 'τάτους', 'τάτη', 'τάτης', 'τάτῃ', 'τάτην', 'άταις', 'τάτας', 'τατα', 'τατά', 'τατε']
	exclamations = ['ὦ', 'Ὦ']
	manner2 = ['ὡς']

	
	other1 = 0
	self2 = 0
	demonstrative3 = 0
	indefinite4 = 0
	personal5 = 0
	reflexive6 = 0
	conjunctions7 = 0
	men8 = 0
	particles9 = 0
	circumstantial10 = 0
	conditional11 = 0
	purpose12 = 0
	mannerA13 = 0
	relative14 = 0
	temporal_causal15 = 0
	inference16 = 0
	mean_relative17 = 0
	interrogative18 = 0
	superlative19 = 0
	exclamation20 = 0
	mannerB21 = 0
	mean_sentence22 = 0


	

	for file in category:
		with open(path+file,"r",encoding='utf-8') as f:
			content = f.read()
			frame_text.append(content)
			frame_category.append(label)
			tokenized = content.split()
			doc_size = len(tokenized)


		for word in tokenized:
			for token in allos_other:
				if word == token:
					other1 += 1

		frame_allos.append(other1/doc_size)


		for word in tokenized:
			for token in autos_self:
				if word == token:
					self2 += 1

		frame_autos.append(self2/doc_size)


		for word in tokenized:
			for token in demonostrative_pronouns:
				if word == token:
					demonstrative3 += 1

		frame_dem.append(demonstrative3/doc_size)


		for word in tokenized:
			for token in indefinite_pronouns:
				if word == token:
					indefinite4 += 1

		frame_indef.append(indefinite4/doc_size)


		for word in tokenized:
			for token in personal_pronouns:
				if word == token:
					personal5 += 1

		frame_pers.append(personal5/doc_size)


		for word in tokenized:
			for token in reflexive_pronouns:
				if word == token:
					reflexive6 += 1

		frame_refl.append(reflexive6/doc_size)


		for word in tokenized:
			for token in coordinating_conjuctions:
				if word == token:
					conjunctions7 += 1

		frame_coord.append(conjunctions7/doc_size)


		for word in tokenized:
			for token in men_indeed:
				if word == token:
					men8 += 1

		frame_men.append(men8/doc_size)


		for word in tokenized:
			for token in particles:
				if word == token:
					particles9 += 1

		frame_particles.append(particles9/doc_size)


		for word in tokenized:
			for token in circumstantial_markers:
				if word == token:
					circumstantial10 += 1

		frame_circum.append(circumstantial10/doc_size)


		for word in tokenized:
			for token in conditional_markers:
				if word == token:
					conditional11 += 1

		frame_cond.append(conditional11/doc_size)


		for word in tokenized:
			for token in purpose:
				if word == token:
					purpose12 += 1

		frame_purp.append(purpose12/doc_size)


		for word in tokenized:
			for token in manner1:
				if word == token:
					mannerA13 += 1

		frame_man1.append(mannerA13/doc_size)


		for word in tokenized:
			for token in relative_clause:
				if word == token:
					relative14 += 1

		frame_rel.append(relative14/doc_size)


		for word in tokenized:
			for token in temporal_causal_markers:
				if word == token:
					temporal_causal15 += 1

		frame_temp_caus.append(temporal_causal15/doc_size)


		for token in relative_clause:
			rel_sentences = re.findall(token+'[\w\s\'᾽]*',content)
			for clauses in rel_sentences:
				mean_relative17 += len(clauses)	#COUNT HOW MANY CHARACTERS IN THE CLAUSES
		frame_mean_rel.append(mean_relative17)



		for word in tokenized:
			for token in superl:
				if word.endswith(token) == True:
					superlative19 += 1
		#print("Total number of superlatives: ", superlatives)
		frame_super.append(superlative19/doc_size)


		for word in tokenized:
			for token in exclamations:
				if word == token:
					exclamation20 += 1

		frame_excl.append(exclamation20/doc_size)



		for word in tokenized:
			for token in manner2:
				if word == token:
					mannerB21 += 1

		frame_man2.append(mannerB21/doc_size)


		segm_sentences = sent_tokenize(content)	#SEGMENT SENTENCES
		total_sentences = len(segm_sentences)	#COUNT NUMBER OF SENTENCES

		no_characters = 0
		for sentence in segm_sentences:
			no_characters += len(sentence)

		mean_sentence22 = no_characters/total_sentences
		frame_length.append(mean_sentence22)


		mean_relative17 = relative14/total_sentences
		frame_mean_rel.append(mean_relative17)



		numQuestion = content.count(question)
		interrogative18 += numQuestion
		#print("Total number of questionmarks: ", questionmarks)
		frame_question.append(interrogative18/total_sentences)


		for sentence in segm_sentences:
			if 'ἤ ὥστε' in sentence:
				pass
			else:
				inference16 += sentence.count('ὥστε')
		frame_inf.append(inference16/total_sentences)



				


features(path_epigram,epigrams,'Epigram')
print('Epigram done')
features(path_hymns,hymns,'Hymn')
print('Hymns done')
features(path_poetry,poetry,'Poetry')
print('Poetry done')
features(path_history,historical,'Historical')
print('Historical done')
features(path_religion,religious,'Religious')
print('Religious done')
#features(path_literature,literature,'Literature')
#print('Literature done')
#features(path_political,political,'Political')
#print('Political done')


data_frame = list(zip(frame_text,
	frame_category,
	frame_allos,
	frame_autos,
	frame_dem,
	frame_indef,
	frame_pers,
	frame_refl,
	frame_coord,
	frame_men,
	frame_particles,
	frame_circum,
	frame_cond,
	frame_purp,
	frame_man1,
	frame_rel,
	frame_temp_caus,
	frame_inf,
	frame_mean_rel,
	frame_question,
	frame_super,
	frame_excl,
	frame_man2,
	frame_length))

df = pd.DataFrame(data_frame,columns = ['Text','Category','Other','Self','Demonstrative','Indefinite','Personal','Reflexive','Conjunctions','Men','Particles','Circumstancial','Conditional','Purpose','Manner1','Relative','Temporal-Causal','Inference','Mean length relative','Interrogative','Superlative','Exclamation','Manner2','Mean length sentence']).set_index('Text', drop=True)
df.to_csv('features_new_test.csv')





