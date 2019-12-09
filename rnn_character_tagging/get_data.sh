#!/usr/bin/env bash
mkdir -p data/austen
cd data

wget http://www.gutenberg.org/files/31100/31100.txt
mv 31100.txt austen/austen.txt

mkdir shakespeare
wget http://www.gutenberg.org/files/100/100-0.txt
mv 100-0.txt shakespeare/shakespeare.txt

git clone https://github.com/scikit-learn/scikit-learn.git
git clone https://github.com/scalaz/scalaz.git