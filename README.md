# Proof of concept multilingual sentiment analysis

## Installation
1. Download and install needed language vectors https://fasttext.cc/docs/en/crawl-vectors.html
2. `pip install -r requirements.txt`

## Sample of languages available
1. Amharic
2. Arabic
3. Bengali
4. English
5. Swahili

## Operation
### Language identification

```
python3 identify.py

import fasttext

PRETRAINED_MODEL_PATH = "vectors/identification/lid.176.bin"
model = fasttext.load_model(PRETRAINED_MODEL_PATH)

sentences = ['Nakutakia mema katika siku yako ya kuzaliwa!']
predictions = model.predict(sentences)
print(predictions)
# returns ([['__label__sw']], [array([0.8830707], dtype=float32)]) Swahili with 88% confidence
```

### Pretrained language sentiment analysis (English)
```
import fasttext
import numpy as np
import scipy
import nltk

PRETRAINED_MODEL_PATH = "vectors/english/cc.en.300.bin"
model = fasttext.load_model(PRETRAINED_MODEL_PATH)


def cos_similarity(sentence, word):
    sent1_emb = model.get_sentence_vector(sentence)
    sent2_emb = model.get_word_vector(word)
    return (1 - scipy.spatial.distance.cosine(sent1_emb, sent2_emb))


good_barometer = "good"
bad_barometer = "bad"

test_good_sentence = "Wow, this is a really great sentence. I love it."
test_bad_sentence = "This is terrible. I hate it."

good_good = cos_similarity(test_good_sentence, good_barometer)
good_bad = cos_similarity(test_good_sentence, bad_barometer)
bad_bad = cos_similarity(test_bad_sentence, bad_barometer)
bad_good = cos_similarity(test_bad_sentence, good_barometer)

print("How good is the test good sentence?", good_good) # 0.5789779424667358
print("How bad is the test good sentence?", good_bad) # 0.4181070625782013
print("Test good sentence is most likely '{}'.".format("good" if good_good > good_bad else "bad")) # 'good'
print("How bad is the test bad sentence?", bad_bad) # 0.41083455085754395
print("How good is the test bad sentence?", bad_good) # 0.39978674054145813
print("Test bad sentence is most likely '{}'.".format("bad" if bad_bad > bad_good else "good")) # 'bad'
```

### Pretrained language sentiment analysis (Swahili)
```
import fasttext
import numpy as np
import scipy
import nltk

PRETRAINED_MODEL_PATH = "vectors/swahili/cc.sw.300.bin"
model = fasttext.load_model(PRETRAINED_MODEL_PATH)


def cos_similarity(sentence, word):
    sent1_emb = model.get_sentence_vector(sentence)
    sent2_emb = model.get_word_vector(word)
    return (1 - scipy.spatial.distance.cosine(sent1_emb, sent2_emb))


good_barometer = "nzuri"
bad_barometer = "mbaya"

test_good_sentence = "Wow, hii ni sentensi ya kushangaza sana. Ninaipenda."
test_bad_sentence = "Sipendi hii. Haiwezekani."

good_good = cos_similarity(test_good_sentence, good_barometer)
good_bad = cos_similarity(test_good_sentence, bad_barometer)
bad_bad = cos_similarity(test_bad_sentence, bad_barometer)
bad_good = cos_similarity(test_bad_sentence, good_barometer)

print("How good is the test good sentence?", good_good) # 0.2866239845752716
print("How bad is the test good sentence?", good_bad) # 0.17274978756904602
print("Test good sentence is most likely '{}'.".format("good" if good_good > good_bad else "bad")) # 'good'
print("How bad is the test bad sentence?", bad_bad) # 0.16904355585575104
print("How good is the test bad sentence?", bad_good) # 0.1387493908405304
print("Test bad sentence is most likely '{}'.".format("bad" if bad_bad > bad_good else "good")) # 'bad'
```
