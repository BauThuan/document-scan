import spacy
from spacy.tokens import DocBin
import pickle
from spacy.lang.vi import Vietnamese

nlp = Vietnamese(use_pyvi=True)

training_data = pickle.load(open('./data/TrainData_Vie_Clean.pickle','rb'))
testing_data = pickle.load(open('./data/TestData_Vie_Clean.pickle','rb'))

db = DocBin()
for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        if span is None:  
                # print(f"Lỗi char_span: ({start}, {end}) - {label} trong văn bản: {text[:50]}...")
                continue  
        ents.append(span)
    print(f"result: {ents}")
    doc.ents = ents
    db.add(doc)
db.to_disk("./data/train.spacy")

db_test = DocBin()
for text, annotations in testing_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        if span is None:  
                # print(f"Lỗi char_span: ({start}, {end}) - {label} trong văn bản: {text[:50]}...")
                continue  
        ents.append(span)
    doc.ents = ents
    db_test.add(doc)
db_test.to_disk("./data/test.spacy")

