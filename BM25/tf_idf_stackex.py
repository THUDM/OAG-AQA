import re
from nltk.stem import PorterStemmer 
import math
import json
from tqdm import tqdm

class QueryParsers:

	def __init__(self, file):
		self.filename = file
		self.query= self.get_queries()

	def get_queries(self):
		q = open(self.filename,'r').read().lower()
		#subsitute all non-word characters with whitespace
		pattern = re.compile('\W+')
		q = pattern.sub(' ', q)
		# split text into words (tokenized list of a document)
		q = q.split()
		# stemming words
		stemmer = PorterStemmer()
		q = [stemmer.stem(w) for w in q ]
		return q

class BuildIndex:
	
	b = 0.75
	k = 1.2
	

	def __init__(self, files, queries_path):
		self.tf = {}
		self.df = {}
		self.filenames = files
		self.docnames=[]
		self.file_to_terms = self.process_files()
		self.regdex = self.regular_index(self.file_to_terms)
		self.invertedIndex = self.inverted_index()
		self.dltable = self.docLtable()
		self.dl = self.docLen()
		self.avgdl = self.avgdocl()
		self.N = self.doc_n()
		self.idf = self.inverse_df()
		# q = QueryParsers('data/query.txt')
		# query = q.query
		# self.total_score  = self.BM25scores(query)
		# self.rankedDocs = self.ranked_docs()

		self.queries,self.labels=self.get_queries(queries_path)
		self.rankedDocs=self.get_queries_ranked_doces(self.queries)

	def process_files(self):
		'''
		input: filenames
		output: a dictionary, with filename as key, and its term list as values 
		'''
		file_to_terms = {}

		# for file in self.filenames:
		# 	#read the whole text of a file into a single string with lowercase
		# 	file_to_terms[file] = open(file,'r').read().lower()
		# 	#subsitute all non-word characters with whitespace
		# 	pattern = re.compile('\W+')
		# 	file_to_terms[file] = pattern.sub(' ', file_to_terms[file])
		# 	# split text into words (tokenized list for a document)
		# 	file_to_terms[file] = file_to_terms[file].split()
		# 	# stemming words
		# 	stemmer = PorterStemmer()
		# 	file_to_terms[file] = [stemmer.stem(w) for w in file_to_terms[file] ]
   
		# with open(self.filenames,'r',encoding='utf8') as fr:
      
		# 	for line in fr.readlines():
       
		# 		line=line.strip()
		# 		line=line.split('\t')
		# 		abstract=line[-2]
		# 		title=line[-1]
		# 		self.docnames.append(title)
    
		# 		file_to_terms[title]=abstract.lower()
		# 		pattern = re.compile('\W+')
		# 		file_to_terms[title] = pattern.sub(' ', file_to_terms[title])
		# 		file_to_terms[title]=file_to_terms[title].split()
		# 		stemmer = PorterStemmer()
		# 		file_to_terms[title] = [stemmer.stem(w) for w in file_to_terms[title] ]

		with open(self.filenames,'r',encoding='utf8') as fr:
			papers=json.loads(fr.read())

		for k,v in tqdm(papers.items(),desc='process file'):
      
			abstract=v['abstract']
			title=k
			self.docnames.append(title)

			file_to_terms[title]=abstract.lower()
			pattern = re.compile('\W+')
			file_to_terms[title] = pattern.sub(' ', file_to_terms[title])
			file_to_terms[title]=file_to_terms[title].split()
			stemmer = PorterStemmer()
			file_to_terms[title] = [stemmer.stem(w) for w in file_to_terms[title] ]
    
		return file_to_terms

	def doc_n(self):
		'''
		return the number of docs in the collection
		'''
		return len(self.file_to_terms)


	def index_one_file(self, termlist):
		'''
		map words to their position for one document
		input: termlist of a document.
		output: a dictionary with word as key, position as value.
		'''
		fileIndex = {}
		for index,word in enumerate(termlist):
			if word in fileIndex.keys():
				fileIndex[word].append(index)
			else:
				fileIndex[word] = [index]

		return fileIndex

	def regular_index(self,termlists):
		'''
		output: a dictionary. key: filename, 
				value: a dictionary with word as key, position as value  
		'''
		regdex = {}

		for filename in termlists.keys():
			regdex[filename] = self.index_one_file(termlists[filename])

		return regdex


	def inverted_index(self):
		'''
		output: dictionary. key: word, 
				value: a dictionary, key is filename, values is its term position of that file
		'''
		total_index = {}
		regdex = self.regdex

		for filename in regdex.keys():
			
			self.tf[filename] = {}

			for word in regdex[filename].keys():
				# tf dict key: filename, value: dict, key is word, value is count
				self.tf[filename][word] = len(regdex[filename][word])
				
				if word in self.df.keys():
					# df dict key: word, value: counts of doc containing that word
					self.df[word] += 1
				else:
					self.df[word] = 1

				if word in total_index.keys():
					if filename in total_index[word].keys():
						total_index[word][filename].extend(regdex[filename][word])
					else:
						total_index[word][filename] = regdex[filename][word]
				else:
					total_index[word] = {filename: regdex[filename][word]}

		return total_index

	def docLtable(self):
		'''
		output: dict, key: word, value: dict(key: number of docs contaiing that word, value:total_freq)
		'''
		dltable = {}
		for w in self.invertedIndex.keys():	
			total_freq = 0
			for file in self.invertedIndex[w].keys():
				total_freq += len(self.invertedIndex[w][file])
			
			dltable[w] = {len(self.invertedIndex[w].keys()):total_freq}
		
		return dltable


	def docLen(self):
		'''
		return a dict, key: filename, value: document length
		'''
		dl = {}
		# for file in self.filenames:
		# 	dl[file]=len(self.file_to_terms[file])
		for name in self.docnames:
			dl[name]=len(self.file_to_terms[name])
		return dl

	def avgdocl(self):
		sum = 0
		for file in self.dl.keys():
			sum += self.dl[file]
		avgdl = sum/len(self.dl.keys())
		return avgdl


	def inverse_df(self):
		'''
		output: inverse doc freq with key: word, value: idf
		'''
		idf = {}
		for w in self.df.keys():
			# idf[w] = math.log((self.N - self.df[w] + 0.5)/(self.df[w] + 0.5))
			idf[w] = math.log((self.N +1 )/self.df[w])
		return idf

    
	def get_score (self,filename,qlist):
		'''
		filename: filename
		qlist: term list of the query 
		output: the score for a document
		'''
		score = 0
		for w in self.file_to_terms[filename]:
			if w not in qlist:
				continue
			wc = len(self.invertedIndex[w][filename])
			score += self.idf[w] * ((wc)* (self.k+1)) / (wc + self.k * 
                                                         (1 - self.b + self.b * self.dl[filename] / self.avgdl))
		return score


	def BM25scores(self,qlist):
		'''
		output: a dictionary with filename as key, score as value
		'''
		total_score = {}
		for doc in self.file_to_terms.keys():
			total_score[doc] = self.get_score(doc,qlist)
		return total_score


	# def ranked_docs(self):
	# 	ranked_docs = sorted(self.total_score.items(), key=lambda x: x[1], reverse=True)[:20]
	# 	return ranked_docs
	def ranked_docs(self,score):
		ranked_docs = sorted(score.items(), key=lambda x: x[1], reverse=True)[:100]
		return ranked_docs

	def get_queries_ranked_doces(self,queries):
		
		res=[]
		for query in tqdm(queries,desc='cacluating BM25'):
			score=self.BM25scores(query)
			res.append(self.ranked_docs(score))
		return res

	def get_queries(self,queries_path):
     
		queries=[]
		labels=[]
		with open(queries_path,'r',encoding='utf8') as fr:
			for line in fr.readlines():
				line=json.loads(line)
				q=line['question'].lower()
				label=line['pids']
    
				pattern = re.compile('\W+')
				q = pattern.sub(' ', q)
				# split text into words (tokenized list of a document)
				q = q.split()
				# stemming words
				stemmer = PorterStemmer()
				q = [stemmer.stem(w) for w in q ]
				queries.append(q)
				labels.append(label)

		return queries,labels

def main():
	oagqa_data_path='/home/shishijie/workspace/project/oag-qa/raw_data/stackex_qa_data/papers_re_stopwords.json'
	queries_path='/home/shishijie/workspace/project/oag-qa/raw_data/stackex_qa_data/qa_test_new.txt'
 
	s=BuildIndex(oagqa_data_path,queries_path)
	rankedDocs=s.rankedDocs
	labels=s.labels
 
	with open('./result/stackex_tfidf/tfidf_recall100.txt','w',encoding='utf8') as fw:
		for line in rankedDocs:
			line=[item[0] for item in line]
			fw.write(",".join(line)+'\n')
	
 

if __name__ =='__main__':
	'''
	query: words selected from 'd5.txt'
	'''
	main()