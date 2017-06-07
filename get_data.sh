#!/usr/bin/env bash
mkdir data
cd data
wget http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz
tar -xvf news20.tar.gz

mkdir glove.6B
cd glove.6B
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
