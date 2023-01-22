
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

new_words = [ "show", "result",  "iv", "one", "two", "previously", "propose", "state","paper","et al","et","al","based"]
stop_words = stop_words.union(new_words)


################################################################
#Function for sorting tf_idf in descending order
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

############################################################
 
def extract_topn_from_vector(doc, topn=10):
    #get_vocabulaire 
    cv=CountVectorizer(stop_words=stop_words, max_features=1000, ngram_range=(1,3))
    X=cv.fit_transform(doc)
    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
    tfidf_transformer.fit(X)
    # get feature names
    feature_names=cv.get_feature_names_out()
    ########################################################
    ####tfidf
    
    tf_idf_vector=tfidf_transformer.transform(cv.transform([doc[0]]))
    #sort the tf-idf vectors by descending order of scores
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results


###################################################################


def tfdd_idf_keys(text):
    k= extract_topn_from_vector(text,topn=10)
    return( list( k.keys()))
 


 

