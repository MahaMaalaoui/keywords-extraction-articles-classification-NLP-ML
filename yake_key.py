import yake
import numpy as np
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

new_words = [ "show", "result",  "iv", "one", "two", "previously", "propose", "state","paper","et al","et","al","based"]
stop_words = stop_words.union(new_words)


def yake_keys(text):

 custom_kw_extractor = yake.KeywordExtractor(lan="en", n=1, dedupLim=0.8, top=10, features=None, stopwords=stop_words)
 keywords = custom_kw_extractor.extract_keywords(text)
 sorted_by_second = sorted(keywords, key=lambda tup: tup[1], reverse=True)
 msg_dict = dict(sorted_by_second)
 k=list(msg_dict.keys())
 return(k)


