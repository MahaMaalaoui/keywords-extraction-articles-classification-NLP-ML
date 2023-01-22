
from collections import Counter
from string import punctuation
import en_core_sci_sm


new_words = [ "show", "result",  "iv", "one", "two", "previously", "propose", "state","paper","et al","et","al","based"]


nlp = en_core_sci_sm.load()

#extract keywords
def get_hotwords(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN'] 
    doc = nlp(text.lower()) 
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation or token.text in new_words):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result

#return keywords
def spacy_keys(text):

 output = set(get_hotwords(text))
 most_common_list = Counter(output).most_common(10)
 key = dict(most_common_list)
 key =list(key.keys())
 return(key)




