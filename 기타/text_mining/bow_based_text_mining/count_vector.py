from nltk.corpus import movie_reviews
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords


def document_features(document, word_features):
    word_count = {}
    for word in document:
        word_count[word] = word_count.get(word, 0) + 1

    features = []
    # word_features의 단어에 대해 계산된 빈도수를 feature에 추가
    for word in word_features:
        features.append(word_count.get(word, 0))        # 빈도가 없는 단어는 0을 입력
    return features


word_features_ex = ['one', 'two', 'teen', 'couples', 'solo']
doc_ex = ['two', 'two', 'couples']
print(document_features(doc_ex, word_features_ex))





