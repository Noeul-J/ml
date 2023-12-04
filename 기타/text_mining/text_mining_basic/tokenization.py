# [데이터 다운로드]
# import nltk
# nltk.download('punkt')
# nltk.download('webtext')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')

# [문장 토큰화]
# from nltk.tokenize import sent_tokenize
#
# para = "Hello everyone. It's good to see you. Let's start our text mining class!"
#
# # 주어진 텍스트를 문장 단위로 토큰화. 주로 . ! ? 등을 이용
# print(sent_tokenize(para))
# para_kor = "안녕하세요, 여러분. 만나서 반갑습니다. 이제 텍스트마이닝 클래스를 시작해봅시다!"
#
# # 한국어에 대해서도 sentence tokenizer는 잘 작동함
# print(sent_tokenize(para_kor))

# import nltk.data
# paragraph_french = """Je t'ai demand si tu m'aimais bien, Tu m'a r pondu none.
# Je t'ai demand si j' tais jolie, Tu m'a r pondu non.
# Je t'ai demand si j' tai dans ton coeur, Tu m'a r pondu non."""
#
# tokenizer = nltk.data.load('tokenizers/punkt/french.pickle')
# print(tokenizer.tokenize(paragraph_french))

# [단어 토큰화]
# 토크나이저의 특성을 파악하고 자신의 목적에 맞는 토크나이저를 선택해야 함
# from nltk.tokenize import word_tokenize
# from nltk.tokenize import WordPunctTokenizer
#
# para = "Hello everyone. It's good to see you. Let's start our text mining class!"
# # 주어딘 text를 word 단위로 tokenize 함
# print(word_tokenize(para))
# # ['Hello', 'everyone', '.', 'It', "'s", 'good', 'to', 'see', 'you', '.', 'Let', "'s", 'start', 'our', 'text', 'mining', 'class', '!']
# print(WordPunctTokenizer().tokenize(para))
# # ['Hello', 'everyone', '.', 'It', "'", 's', 'good', 'to', 'see', 'you', '.', 'Let', "'", 's', 'start', 'our', 'text', 'mining', 'class', '!']
#
# para_kor = "안녕하세요, 여러분. 만나서 반갑습니다. 이제 텍스트마이닝 클래스를 시작해봅시다!"
# print(word_tokenize(para_kor))

# [정규표현식을 이용한 토큰화]
# import re
# # re.findall - 문자열에서 검색해서 매칭되는 모든 값을 반환
# print(re.findall("[abc]", "How are you, boy?"))
# print(re.findall("[0123456789]", "3a7b5c9d"))
# print(re.findall("[\w]", "3a 7b_ '.^&5c9d"))        # [a-zA-Z0-9_] -- ['3', 'a', '7', 'b', '_', '5', 'c', '9', 'd']
# print(re.findall("[_]+", "a_b, c__d, e___f"))       # ['_', '__', '___'] -- + 한 번 이상 반복된 부분
# print(re.findall("[\w]+","How are you, boy"))       # \w에는 공백(스페이스)이 포함되지 X -- ['How', 'are', 'you', 'boy']
# print(re.findall("[o]{2,4}", "oh, hoow are yoooou, boooooooy?"))    # o가 2~4회 반복된 문자열 --
# # ['oo', 'oooo', 'oooo', 'ooo']

# # [nltk에서 정규 표현식]
# from nltk.tokenize import RegexpTokenizer
#
# # regular expression(정규식)을 이용한 tokenizer
#
# text = "Sorry, I can't go there."
#
# # 단어 단위로 tokenize \w:문자나 숫자를 의미. 즉 문자나 숫자 혹은 '가 반복되는 것을 찾아냄
# tokenizer = RegexpTokenizer("[\w']+")
#
# # can't를 하나의 단어로 인식
# print(tokenizer.tokenize(text))                 # ['Sorry', 'I', "can't", 'go', 'there']
#
# tokenizer = RegexpTokenizer("[\w']{3,}")        # 세 글자 이상의 단어들만
# print(tokenizer.tokenize(text.lower()))         # ['sorry', "can't", 'there']

# # 불용어 제거
# from nltk.tokenize import RegexpTokenizer
# from nltk.corpus import stopwords   # 일반적으로 분석대상이 아닌 단어들
# english_stops = set(stopwords.words('english'))     # 반복이 되지 않도록 set으로 변환
# print(english_stops)
#
# text = "Sorry, I couldn't go to movie yesterday."
#
# tokenizer = RegexpTokenizer("[\w']+")
# tokens = tokenizer.tokenize(text.lower())   # word_tokenize 로 토큰화
#
# # stopwords 를 제외한 단어들만으로 list를 생성
# result = [word for word in tokens if word not in english_stops]
#
# print(result)       # ['sorry', 'go', 'movie', 'yesterday']
#
# # 자신만의 불용어 사전을 만들 수 있음
# # 한글 처리에도 유용함
# my_stopword = ['i', 'go', 'to']
# result = [word for word in tokens if word not in my_stopword]
# print(result)       # ['sorry', "couldn't", 'movie', 'yesterday']

# # [어간추출] - 포터 스테머
# # - 모든 단어가 같은 규칙에 따라 변환됨
# from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
#
# stemmer = PorterStemmer()
# print(stemmer.stem('cooking'), stemmer.stem('cookery'), stemmer.stem('cookbooks'))
# # cook cookeri cookbook
#
# # 토큰화와 결합해 어간 추출
# para = "Hello everyone. It's good to see you. Let's start our text mining class!"
# tokens = word_tokenize(para)
# print(tokens)   # ['Hello', 'everyone', '.', 'It', "'s", 'good', 'to', 'see', 'you', '.', 'Let', "'s", 'start', 'our', 'text', 'mining', 'class', '!']
# result = [stemmer.stem(token) for token in tokens]  # 모든 토큰에 대해 스테밍 실행
# print(result)   # ['hello', 'everyon', '.', 'it', "'s", 'good', 'to', 'see', 'you', '.', 'let', "'s", 'start', 'our', 'text', 'mine', 'class', '!']

# # [어간 추출] - 랭카스터 스테머
# # from nltk.stem import LancasterStemmer
# # stemmer = LancasterStemmer()
# # print(stemmer.stem('cooking'), stemmer.stem('cookery'), stemmer.stem('cookbooks'))
# # # cook cookery cookbook

# [표제어 추출]
# # import nltk
# # nltk.download('omw-1.4')
#
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# print(lemmatizer.lemmatize('cooking'))
# print(lemmatizer.lemmatize('cooking', pos='v'))     # 품사를 지정
# print(lemmatizer.lemmatize('cookery'))
# print(lemmatizer.lemmatize('cookbooks'))

# # 품사 확인
# # lemmatizing 과 stemming 비교
# from nltk.stem import PorterStemmer
# from nltk.stem import WordNetLemmatizer
#
# stemmer = PorterStemmer()
# lemmatizer = WordNetLemmatizer()
# print('stemming result:', stemmer.stem('believes'))
# print('lemmatizing result:', lemmatizer.lemmatize('believes'))
# print('lemmatizing result:', lemmatizer.lemmatize('believes', pos='v'))

# # [품사 태깅]
# import nltk
# from nltk.tokenize import word_tokenize
# # nltk.download('tagsets')
#
# tokens = word_tokenize("Hello everyone. It's good to see you. Let's start our text mining class!")
# # print(nltk.pos_tag(tokens))
# # [('Hello', 'NNP'), ('everyone', 'NN'), ('.', '.'), ('It', 'PRP'), ("'s", 'VBZ'), ('good', 'JJ'),
# # ('to', 'TO'), ('see', 'VB'), ('you', 'PRP'), ('.', '.'), ('Let', 'VB'), ("'s", 'POS'), ('start', 'VB'),
# # ('our', 'PRP$'), ('text', 'NN'), ('mining', 'NN'), ('class', 'NN'), ('!', '.')]
#
# # 품사 약어의 의미와 설명을 보고 싶을 때
# # nltk.help.upenn_tagset('CC')
# # CC: conjunction, coordinating
# #     & 'n and both but either et for less minus neither nor or plus so
# #     therefore times v. versus vs. whether yet
#
# my_tag_set = ['NN', 'VB', 'JJ']
# my_words = [word for word, tag in nltk.pos_tag(tokens) if tag in my_tag_set]
# print(my_words)
# # ['everyone', 'good', 'see', 'Let', 'start', 'text', 'mining', 'class']
#
# words_with_tag = ['/'.join(item) for item in nltk.pos_tag(tokens)]
# print(words_with_tag)
# # ['Hello/NNP', 'everyone/NN', './.', 'It/PRP', "'s/VBZ", 'good/JJ', 'to/TO', 'see/VB',
# # 'you/PRP', './.', 'Let/VB', "'s/POS", 'start/VB', 'our/PRP$', 'text/NN', 'mining/NN', 'class/NN', '!/.']

# 한글 형태소 분석과 품사 태깅
# import nltk
# from nltk import word_tokenize
#
# sentence = ''' 절망의 반대가 희망은 아니다.
# 어두운 밤하늘에 별이 빛나듯
# 희망은 절망 속에 싹트는 거지
# 만약에 우리가 희망함이 적다면
# 그 누가 세상을 비출어줄까
# 정희성, 희망 공부'''
#
# tokens = word_tokenize(sentence)
# print(tokens)
# print(nltk.pos_tag(tokens))
# # ['절망의', '반대가', '희망은', '아니다', '.', '어두운', '밤하늘에', '별이', '빛나듯', '희망은', '절망', '속에', '싹트는', '거지',
# # '만약에', '우리가', '희망함이', '적다면', '그', '누가', '세상을', '비출어줄까', '정희성', ',', '희망', '공부']
# # [('절망의', 'JJ'), ('반대가', 'NNP'), ('희망은', 'NNP'), ('아니다', 'NNP'), ('.', '.'), ('어두운', 'VB'), ('밤하늘에', 'JJ'),
# # ('별이', 'NNP'), ('빛나듯', 'NNP'), ('희망은', 'NNP'), ('절망', 'NNP'), ('속에', 'NNP'), ('싹트는', 'NNP'), ('거지', 'NNP'),
# # ('만약에', 'NNP'), ('우리가', 'NNP'), ('희망함이', 'NNP'), ('적다면', 'NNP'), ('그', 'NNP'), ('누가', 'NNP'),
# # ('세상을', 'NNP'), ('비출어줄까', 'NNP'), ('정희성', 'NNP'), (',', ','), ('희망', 'NNP'), ('공부', 'NNP')]

from konlpy.tag import Okt

sentence = ''' 절망의 반대가 희망은 아니다.
어두운 밤하늘에 별이 빛나듯
희망은 절망 속에 싹트는 거지
만약에 우리가 희망함이 적다면
그 누가 세상을 비출어줄까
정희성, 희망 공부'''

t = Okt()
print('형태소:', t.morphs(sentence))
print()
print('명사:', t.nouns(sentence))

# 형태소: ['절망', '의', '반대', '가', '희망', '은', '아니다', '.', '\n', '어', '두운', '밤하늘', '에', '별', '이', '빛나듯',
# '\n', '희망', '은', '절망', '속', '에', '싹트는', '거지', '\n', '만약', '에', '우리', '가', '희망', '함', '이', '적다면',
# '\n', '그', '누가', '세상', '을', '비출어줄까', '\n', '정희성', ',', '희망', '공부']
#
# 명사: ['절망', '반대', '희망', '어', '두운', '밤하늘', '별', '희망', '절망', '속', '거지', '만약',
# '우리', '희망', '함', '그', '누가', '세상', '정희성', '희망', '공부']

print('품사 태깅 결과:', t.pos(sentence))
# [('절망', 'Noun'), ('의', 'Josa'), ('반대', 'Noun'), ('가', 'Josa'), ('희망', 'Noun'), ('은', 'Josa'),
# ('아니다', 'Adjective'), ('.', 'Punctuation'), ('\n', 'Foreign'), ('어', 'Noun'), ('두운', 'Noun'),
# ('밤하늘', 'Noun'), ('에', 'Josa'), ('별', 'Noun'), ('이', 'Josa'), ('빛나듯', 'Verb'), ('\n', 'Foreign'),
# ('희망', 'Noun'), ('은', 'Josa'), ('절망', 'Noun'), ('속', 'Noun'), ('에', 'Josa'), ('싹트는', 'Verb'),
# ('거지', 'Noun'), ('\n', 'Foreign'), ('만약', 'Noun'), ('에', 'Josa'), ('우리', 'Noun'), ('가', 'Josa'),
# ('희망', 'Noun'), ('함', 'Noun'), ('이', 'Josa'), ('적다면', 'Verb'), ('\n', 'Foreign'), ('그', 'Noun'),
# ('누가', 'Noun'), ('세상', 'Noun'), ('을', 'Josa'), ('비출어줄까', 'Verb'), ('\n', 'Foreign'), ('정희성', 'Noun'),
# (',', 'Punctuation'), ('희망', 'Noun'), ('공부', 'Noun')]



