python score.py --maxevals 50 --save
python score.py --maxevals 50 --with_bigrams --save
python score.py --packg spacy --maxevals 50 --save
python score.py --packg spacy --maxevals 50 --with_bigrams --save

python score.py --algo lda --n_topics 20 --with_cv --save
python score.py --algo lda --n_topics 50 --with_cv --save
python score.py --algo lda --n_topics 20 --with_bigrams --with_cv --save
python score.py --algo lda --n_topics 50 --with_bigrams --with_cv --save
python score.py --packg spacy --algo lda --n_topics 20 --with_cv --save
python score.py --packg spacy --algo lda --n_topics 50 --with_cv --save
python score.py --packg spacy --algo lda --n_topics 20 --with_bigrams --with_cv --save
python score.py --packg spacy --algo lda --n_topics 50 --with_bigrams --with_cv --save

python score.py --algo ensemb --n_topics 20 --with_cv --save
python score.py --algo ensemb --n_topics 20 --with_bigrams --with_cv --save
python score.py --packg spacy --algo ensemb --n_topics 20 --with_cv --save
python score.py --packg spacy --algo ensemb --n_topics 20 --with_bigrams --with_cv --save