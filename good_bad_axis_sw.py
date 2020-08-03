import fasttext
import numpy as np
import scipy
import nltk


stop = list()
with open("stopwords/swahili/stopwords.txt", "r") as sw_file:
    stop = sw_file.readlines()

stop = [word.strip() for word in stop]


PRETRAINED_MODEL_PATH = "vectors/swahili/cc.sw.300.bin"
model = fasttext.load_model(PRETRAINED_MODEL_PATH)


def cos_similarity(data_1, data_2):
    sent1_emb = np.mean([model[x] for word in data_1 for x in word.split() if x not in stop], axis=0)
    sent2_emb = np.mean([model[x] for word in data_2 for x in word.split() if x not in stop], axis=0)
    return (1 - scipy.spatial.distance.cosine(sent1_emb, sent2_emb))


good_barometer = ["nzuri"]
bad_barometer = ["mbaya"]

test_good_sentence = ["Wow, hii ni sentensi ya kushangaza sana. Ninaipenda."]
test_bad_sentence = ["Sipendi hii. Haiwezekani."]

good_good = cos_similarity(test_good_sentence, good_barometer)
good_bad = cos_similarity(test_good_sentence, bad_barometer)
bad_bad = cos_similarity(test_bad_sentence, bad_barometer)
bad_good = cos_similarity(test_bad_sentence, good_barometer)

print("How good is the test good sentence?", good_good)
print("How bad is the test good sentence?", good_bad)
print("Test good sentence is most likely '{}'.".format("good" if good_good > good_bad else "bad"))
print("How bad is the test bad sentence?", bad_bad)
print("How good is the test bad sentence?", bad_good)
print("Test bad sentence is most likely '{}'.".format("bad" if bad_bad > bad_good else "good"))