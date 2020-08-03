import fasttext

PRETRAINED_MODEL_PATH = "vectors/identification/lid.176.bin"
model = fasttext.load_model(PRETRAINED_MODEL_PATH)

sentences = ['Nakutakia mema katika siku yako ya kuzaliwa!']
predictions = model.predict(sentences)
print(predictions)
