from tqdm import tqdm
import pandas as pd

#Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import TfidfModel

# nltk
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# spacy
import spacy
from spacy.cli import download
download("de_core_news_sm")

#vis
import pyLDAvis
import pyLDAvis.gensim_models


stopwords = stopwords.words("german")
news = pd.read_csv("KoeZ_1866.csv", sep=';')

data = news['text'].tolist()

def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
    texts_out = []
    for text in tqdm(texts):
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return (texts_out)


lemmatized_texts = lemmatization(data)
print (lemmatized_texts[0][:1000])

def gen_words(texts):
    final = []
    for text in tqdm(texts):
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return (final)

data_words = gen_words(lemmatized_texts)

print (data_words[0][0:20])

#BIGRAMS AND TRIGRAMS
bigram_phrases = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram_phrases = gensim.models.Phrases(bigram_phrases[data_words], threshold=100)

bigram = gensim.models.phrases.Phraser(bigram_phrases)
trigram = gensim.models.phrases.Phraser(trigram_phrases)

def make_bigrams(texts):
    return([bigram[doc] for doc in texts])

def make_trigrams(texts):
    return ([trigram[bigram[doc]] for doc in texts])

data_bigrams = make_bigrams(data_words)
data_bigrams_trigrams = make_trigrams(data_bigrams)

print (data_bigrams_trigrams[0][0:20])

#TF-IDF REMOVAL
id2word = corpora.Dictionary(data_bigrams_trigrams)

texts = data_bigrams_trigrams

corpus = [id2word.doc2bow(text) for text in texts]
# print (corpus[0][0:20])

tfidf = TfidfModel(corpus, id2word=id2word)

low_value = 0.03
words  = []
words_missing_in_tfidf = []
for i in tqdm(range(0, len(corpus))):
    bow = corpus[i]
    low_value_words = [] #reinitialize to be safe. You can skip this.
    tfidf_ids = [id for id, value in tfidf[bow]]
    bow_ids = [id for id, value in bow]
    low_value_words = [id for id, value in tfidf[bow] if value < low_value]
    drops = low_value_words+words_missing_in_tfidf
    for item in drops:
        words.append(id2word[item])
    words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids] # The words with tf-idf socre 0 will be missing

    new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
    corpus[i] = new_bow

lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus[:-1],
                                           id2word=id2word,
                                           num_topics=, # enter number of topics here
                                           random_state=42,
                                           update_every=1,
                                           chunksize=2000,
                                           passes=10,
                                           alpha="auto",
                                           eval_every=None)

test_doc = corpus[-1]

vector = lda_model[test_doc]
print (vector)

lda_model.show_topics(num_topics=*) # number of topics to show must not be larger thant the number of topics generated

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=30)


pyLDAvis.save_html(vis, '*.html') # define export name

def find_topics(ldamodel, topic):
    wp = ldamodel.show_topic(topic[0])
    keywords = ", ".join([word for word, prop in wp[:5]])
    topic_num = int(topic[0])
    topic_percentage = round(topic[1], 4)
    return keywords, topic_num, topic_percentage

main_topic = {}
second_topic = {}
third_topic = {}
main_percentage = {}
second_percentage = {}
third_percentage = {}
main_keywords = {}
second_keywords = {}
third_keywords = {}
text_snippets = {}

for i, topic_list in enumerate(ldamodel[corpus]):
    topic_list = sorted(topic_list, key=lambda x: (x[1]), reverse=True)

    try:
        main_keywords[i], main_topic[i], main_percentage[i] = find_topics(ldamodel, topic_list[0])
    except IndexError:
        main_keywords[i], main_topic[i], main_percentage[i] = "None", "None", "None"

    try:
        second_keywords[i], second_topic[i], second_percentage[i] = find_topics(ldamodel, topic_list[1])
    except IndexError:
        second_keywords[i], second_topic[i], second_percentage[i] = "None", "None", "None"

    try:
        third_keywords[i], third_topic[i], third_percentage[i] = find_topics(ldamodel, topic_list[2])
    except IndexError:
        third_keywords[i], third_topic[i], third_percentage[i] = "None", "None", "None"

    text_snippets[i] = text[i][:8]

news['Main Topic'] = main_topic
news['Main Topic Percentage'] = main_percentage
news['Main Keywords'] = main_keywords
news['Second Topic'] = second_topic
news['Second Topic Percentage'] = second_percentage
news['Second Keywords'] = second_keywords
news['Third Topic'] = third_topic
news['Third Topic Percentage'] = third_percentage
news['Third Keywords'] = third_keywords
news['Text Snippets'] = text_snippets

news.to_csv('*.csv', sep=';') # name the export file
