import nltk
# nltk.download('gutenberg')
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords   # 일반적으로 분석대상이 아닌 단어들
import matplotlib.pyplot as plt

# file_names = gutenberg.fileids()
# # 파일 제목을 읽어온다.
# print(file_names)

doc_alice = gutenberg.open('carroll-alice.txt').read()
# print('#Num of characters used:', len(doc_alice))   # 사용된 문자의 수
# print('#Text sample:')
# print(doc_alice[:500])  # 앞의 500 자만 출력

# tokens_alice = word_tokenize(doc_alice) # 토큰화 실행

# print('#Num of tokens used:', len(tokens_alice))
# print('#Token sample:')
# print(tokens_alice[:20])

# #Num of tokens used: 33494
# #Token sample:
# ['[', 'Alice', "'s", 'Adventures', 'in', 'Wonderland', 'by', 'Lewis', 'Carroll', '1865', ']', 'CHAPTER', 'I',
# '.', 'Down', 'the', 'Rabbit-Hole', 'Alice', 'was', 'beginning']

# [포터 스테머로 스테밍 하고 토큰 확인]
# stemmer = PorterStemmer()
#
# # 모든 토큰에 대해 스테밍 실행
# stem_tokens_alice = [stemmer.stem(token) for token in tokens_alice]
#
# print('#Num of tokens after stemming:', len(stem_tokens_alice))
# print('#Toekn sample:')
# print(stem_tokens_alice[:20])
#
# # #Num of tokens after stemming: 33494
# # #Toekn sample:
# # ['[', 'alic', "'s", 'adventur', 'in', 'wonderland', 'by', 'lewi', 'carrol', '1865', ']',
# # 'chapter', 'i', '.', 'down', 'the', 'rabbit-hol', 'alic', 'wa', 'begin']

# [WordNetLemmatizer 이용해 표제어 추출]
# lemmatizer = WordNetLemmatizer()
#
# # 모든 토큰에 대해 스테밍 실행
# lem_tokens_alice = [lemmatizer.lemmatize(token) for token in tokens_alice]
#
# print('#Num of tokens after lemmatization:', len(lem_tokens_alice))
# print('#Token sample:')
# print(lem_tokens_alice[:20])
#
# # #Num of tokens after lemmatization: 33494
# # #Token sample:
# # ['[', 'Alice', "'s", 'Adventures', 'in', 'Wonderland', 'by', 'Lewis', 'Carroll', '1865', ']',
# # 'CHAPTER', 'I', '.', 'Down', 'the', 'Rabbit-Hole', 'Alice', 'wa', 'beginning']

# [정규표현식을 이용한 토큰화]
tokenizer = RegexpTokenizer("[\w']{3,}")

reg_tokens_alice = tokenizer.tokenize(doc_alice.lower())
# print('#Num of tokens with RegexpTokenizer:', len(reg_tokens_alice))
# print('#Token sample:')
# print(reg_tokens_alice[:20])
#
# # #Num of tokens with RegexpTokenizer: 21616
# # #Token sample:
# # ["alice's", 'adventures', 'wonderland', 'lewis', 'carroll', '1865', 'chapter', 'down',
# # 'the', 'rabbit', 'hole', 'alice', 'was', 'beginning', 'get', 'very', 'tired', 'sitting', 'her', 'sister']

english_stops = set(stopwords.words('english'))      # 반복되지 않게 set으로 변환

# stopwords를 제외한 단어들만으로 리스트를 생성
result_alice = [word for word in reg_tokens_alice if word not in english_stops]

# print('#Num of tokens after elimination:', len(result_alice))
# print('#Token sample:')
# print(result_alice[:20])

# #Num of tokens after elimination: 12999
# #Token sample:
# ["alice's", 'adventures', 'wonderland', 'lewis', 'carroll', '1865', 'chapter', 'rabbit',
# 'hole', 'alice', 'beginning', 'get', 'tired', 'sitting', 'sister', 'bank', 'nothing', 'twice', 'peeped', 'book']

# [각 단어별 빈도 계산]
alice_word_count = dict()
for word in result_alice:
    alice_word_count[word] = alice_word_count.get(word, 0) + 1

# print('#Num of used words:', len(alice_word_count))

sorted_word_count = sorted(alice_word_count, key=alice_word_count.get, reverse=True)

# print('#Top 20 high frequency words:')
# for key in sorted_word_count[:20]:      # 빈도수 상위 20개의 단어를 출력
#     print(f'{repr(key)}: {alice_word_count[key]}', end=',')

# #Num of used words: 2687
# #Top 20 high frequency words:
# 'said': 462,'alice': 385,'little': 128,'one': 98,'know': 88,'like': 85,'went': 83,'would': 78,
# 'could': 77,'thought': 74,'time': 71,'queen': 68,'see': 67,'king': 61,'began': 58,'turtle': 57,
# "'and": 56,'way': 56,'mock': 56,'quite': 55,

# 정렬된 단어 리스트에 대해 빈도수를 가져와서 리스트를 생성
# w = [alice_word_count[key] for key in sorted_word_count]
#
# plt.plot(w)
# plt.show()

n = sorted_word_count[:20][::-1]    # 빈도수 상위 20개의 단어를 추출해 역순으로 정렬
w = [alice_word_count[key] for key in n]    # 20개 단어에 대한 빈도
plt.barh(range(len(n)), w, tick_label=n)    # 수평 막대 그래프
plt.show()
