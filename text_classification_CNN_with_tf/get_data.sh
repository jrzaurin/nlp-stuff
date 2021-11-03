#!/usr/bin/env bash
mkdir 20_newsgroup
cd 20_newsgroup
wget http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz
tar -xvf news20.tar.gz

mkdir glove.6B
cd glove.6B
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
