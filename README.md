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
Language identification

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

Untrained language sentiment analysis
```
import fasttext
import numpy as np
import scipy
from nltk.corpus import stopwords

stop = stopwords.words('english')

PRETRAINED_MODEL_PATH = "vectors/english/cc.en.300.bin"
model = fasttext.load_model(PRETRAINED_MODEL_PATH)


def cos_similarity(data_1, data_2):
    sent1_emb = np.mean([model[x] for word in data_1 for x in word.split() if x not in stop], axis=0)
    sent2_emb = np.mean([model[x] for word in data_2 for x in word.split() if x not in stop], axis=0)
    return (1 - scipy.spatial.distance.cosine(sent1_emb, sent2_emb))


good_barometer = ["good"]
bad_barometer = ["bad"]

test_good_sentence = ["Wow, this is a really great sentence. I love it."]
test_bad_sentence = ["This is terrible. I hate it."]

good_good = cos_similarity(test_good_sentence, good_barometer)
good_bad = cos_similarity(test_good_sentence, bad_barometer)
bad_bad = cos_similarity(test_bad_sentence, bad_barometer)
bad_good = cos_similarity(test_bad_sentence, good_barometer)

print("How good is the test good sentence?", good_good) # 0.4440871477127075
print("How bad is the test good sentence?", good_bad) # 0.3507806658744812
print("Test good sentence is most likely '{}'.".format("good" if good_good > good_bad else "bad")) # 'good'
print("How bad is the test bad sentence?", bad_bad) # 0.35548049211502075
print("How good is the test bad sentence?", bad_good) # 0.3502688705921173
print("Test bad sentence is most likely '{}'.".format("bad" if bad_bad > bad_good else "bad")) # 'bad'
```
