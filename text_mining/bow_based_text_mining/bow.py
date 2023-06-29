# import nltk
# nltk.download('movie_g
from nltk.corpus import movie_reviews

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