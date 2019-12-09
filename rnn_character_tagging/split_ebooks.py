import re
import os

authors = ['austen', 'shakespeare']

ebook_d = {}
ebook_d['austen'] = {}
ebook_d['shakespeare'] = {}

ebook_d['austen']['dir'] = 'data/austen_clean'
ebook_d['austen']['fname'] = 'austen.txt'
ebook_d['austen']['regex'] = 'Chapter\s+.*|CHAPTER\s+.*'
ebook_d['austen']['startidx'] = 1
ebook_d['austen']['endex'] = 'THE END'

ebook_d['shakespeare']['dir'] = 'data/shakespeare_clean'
ebook_d['shakespeare']['fname'] = 'shakespeare.txt'
ebook_d['shakespeare']['regex'] = '\s+\d+\s+|ACT\s+.*\.|SCENE\s+.*\.'
ebook_d['shakespeare']['startidx'] = 3
ebook_d['shakespeare']['endex'] = 'FINIS'

for author in authors:
	filepath = os.path.join(ebook_d[author]['dir'],ebook_d[author]['fname'])
	with open(filepath, 'r') as f:
	    ebook = f.read()
	f.close()

	endex = ebook_d[author]['endex']
	startidx = ebook_d[author]['startidx']
	the_end = [m.start() for m in re.finditer(endex, ebook)][-1]
	ebook = ebook[:the_end]
	parts = re.split(ebook_d[author]['regex'], ebook)[startidx:]

	for i,p in enumerate(parts):
		fname = 'part' + str(i).zfill(4) + '.txt'
		fpath = os.path.join(ebook_d[author]['dir'],fname)
		with open(fpath, 'w') as f:
			f.write(p)
		f.close()
	os.remove(filepath)
