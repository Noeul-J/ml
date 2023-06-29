from nltk.corpus import gutenberg
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

doc_alice = gutenberg.open('carroll-alice.txt').read()

tokenizer = RegexpTokenizer("[\w']{3,}")
reg_tokens_alice = tokenizer.tokenize(doc_alice.lower())

english_stops = set(stopwords.words('english'))
result_alice = [word for word in reg_tokens_alice if word not in english_stops]

# [각 단어별 빈도 계산]
alice_word_count = dict()
for word in result_alice:
    alice_word_count[word] = alice_word_count.get(word, 0) + 1

# # 워드 클라우드 이미지 생성
# # wordcloud = WordCloud().generate(doc_alice)
# wordcloud = WordCloud(max_font_size=60).generate_from_frequencies(alice_word_count)
#
# plt.figure()
# plt.axis('off')
# plt.imshow(wordcloud, interpolation='bilinear')       # 이미지를 출력
# plt.show()

# wordcloud.to_array().shape

# 배경이미지를 불러와서 넘파이 array로 변환
alice_mask = np.array(Image.open("alice_mask.png"))

wc = WordCloud(background_color="white",    # 배경색 지정
               max_words=30,                # 출력할 최대 단어 수
               mask=alice_mask,             # 배경으로 사용할 이미지
               contour_width=3,             # 테두리 굵기
               contour_color='steelblue')   # 테두리 색

wc.generate_from_frequencies(alice_word_count)      # 워드 클라우드 생성
wc.to_file("alice.png")                             # 결과를 이미지 파일로 저장

# 화면에 결과를 출력
plt.figure()
plt.axis("off")
plt.imshow(wc, interpolation='bilinear')
plt.show()