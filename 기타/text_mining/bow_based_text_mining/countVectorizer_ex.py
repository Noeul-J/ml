from nltk.corpus import movie_reviews

# data 준비, movie_reviews.raw() 사용해 raw text를 추출
reviews = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
