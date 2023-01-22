import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer




lem = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

new_words = [ "show", "result",  "iv", "one", "two", "previously", "propose", "state","paper","et al","et","al","based"]
stop_words = stop_words.union(new_words)


def normalise (corpus,x):
    
    #Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', x )
    
    #Convert to lowercase
    text = text.lower()
    
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    ##Convert to list from string
    text = text.split()
    
    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in  
            stop_words] 
    text = " ".join(text)
    corpus.append(text)
