import spacy
from spacy.training.example import Example

def train_ner(train_data, model=None, iterations=20):
    if model:
        nlp = spacy.load(model)
    else:
        nlp = spacy.blank("en")

    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    for _, annotations in train_data:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])

    optimizer = nlp.begin_training()

    for i in range(iterations):
        for text, annotations in train_data:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.3, sgd=optimizer)

    return nlp