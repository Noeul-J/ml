# import nltk
# nltk.download('movie_reviews')
from nltk.corpus import movie_reviews
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# print('#review count:', len(movie_reviews.fileids()))       # 영화 리뷰 문서의 id를 반환
# print('#samples of file ids:', movie_reviews.fileids()[:10])        # id를 10개 까지만 출력
#
# fileid = movie_reviews.fileids()[0]     # 첫번째 문서의 id를 반환
# print('#id of the first review:', fileid)
#
# # 첫번째 문서의 내용을 200자 까지만 추출
# print('#first review content:\n', movie_reviews.raw(fileid)[:200])
#
# # 첫번째 문서를 sentence tokenize 한 결과 중 앞 두문장
# print('#sentence tokenization result:', movie_reviews.sents(fileid)[:2])
#
# # 첫번째 문서를 word tokenize 한 결과 중 앞 20개 단어
# print('#word tokenization result:', movie_reviews.words(fileid)[:20])
#
# # #review count: 2000
# # #samples of file ids: ['neg/cv000_29416.txt', 'neg/cv001_19502.txt', 'neg/cv002_17424.txt', 'neg/cv003_12683.txt',
# # 'neg/cv004_12641.txt', 'neg/cv005_29357.txt', 'neg/cv006_17022.txt', 'neg/cv007_4992.txt',
# # 'neg/cv008_29326.txt', 'neg/cv009_29417.txt']
# # #id of the first review: neg/cv000_29416.txt
# # #first review content:
# #  plot : two teen couples go to a church party , drink and then drive .
# # they get into an accident .
# # one of the guys dies , but his girlfriend continues to see him in her life , and has nightmares .
# # w
# # #sentence tokenization result: [['plot', ':', 'two', 'teen', 'couples', 'go', 'to', 'a', 'church', 'party', ',',
# # 'drink', 'and', 'then', 'drive', '.'], ['they', 'get', 'into', 'an', 'accident', '.']]
# # #word tokenization result: ['plot', ':', 'two', 'teen', 'couples', 'go', 'to', 'a', 'church', 'party', ',', 'drink',
# # 'and', 'then', 'drive', '.', 'they', 'get', 'into', 'an']

documents = [list(movie_reviews.words(fileid)) for fileid in movie_reviews.fileids()]
print(documents[0][:50])        # 첫째 문서의 앞 50개 단어를 출력

word_count = {}
for text in documents:
    for word in text:
        word_count[word] = word_count.get(word, 0) + 1

sorted_features = sorted(word_count, key=word_count.get, reverse=True)
for word in sorted_features[:10]:
    print(f"count of '{word}': {word_count[word]}", end=',')

tokenizer = RegexpTokenizer("[\w']{3,}")
english_stops = set(stopwords.words('english'))

# words() 대신 raw()로 원문을 가져옴
documents = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]

# stopwords의 적용과 토큰화를 동시에 수행
tokens = [[token for token in tokenizer.tokenize(doc) if token not in english_stops] for doc in documents]
word_count = {}
for text in tokens:
    for word in text:
        word_count[word] = word_count.get(word, 0) + 1

sorted_features = sorted(word_count, key=word_count.get, reverse=True)

print('num of features:', len(sorted_features))
for word in sorted_features[:10]:
    print(f"count of '{word}': {word_count[word]}", end=',')

# 빈도가 높은 상위 1000개의 단어만 추출해 features를 구성
word_features = sorted_features[:1000]
