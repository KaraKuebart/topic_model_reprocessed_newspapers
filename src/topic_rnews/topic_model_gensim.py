import gensim
import spacy
from gensim import corpora
from gensim.models import TfidfModel
import numpy as np

import read_data
import pandas as pd
from tqdm import tqdm
import datetime


def gensim_lda(data:pd.DataFrame, num_topics:int=None) -> pd.DataFrame:
    num_docs = data.shape[0]
    if num_topics is None:
        num_topics = round(np.power(num_docs, 1 / 6))
    text_data = data['text'].tolist()
    lemmatized_texts = lemmatization(text_data)
    data_words = gen_words(lemmatized_texts)

    # BIGRAMS AND TRIGRAMS
    data_bigrams_trigrams = make_bigrams_trigrams(data_words)

    # TF-IDF REMOVAL
    id2word = corpora.Dictionary(data_bigrams_trigrams)
    texts = data_bigrams_trigrams
    corpus = [id2word.doc2bow(text) for text in texts]
    tfidf = TfidfModel(corpus, id2word=id2word)

    # reduce corpus
    corpus = reduce_corpus(corpus, id2word, tfidf)

    # lda_model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus[:-1],
                                                id2word=id2word,
                                                num_topics=num_topics,  # enter number of topics here
                                                random_state=42,
                                                update_every=1,
                                                chunksize=2000,
                                                passes=10,
                                                alpha="auto",
                                                eval_every=None)

    lda_model.show_topics(
        num_topics=5)  # number of topics to show must not be larger thant the number of topics generated

    data = export_results(corpus, data, lda_model, texts)

    return data


def lemmatization(texts_in, allowed_postags=None):
    if allowed_postags is None:
        allowed_postags = ["NOUN", "ADJ", "VERB", "ADV"]
    nlp = spacy.load("de_core_news_sm", disable=["parser", "ner"])
    texts_out = []
    for text in tqdm(texts_in):
        text = str(text)
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    return texts_out


def gen_words(texts_in):
    final = []
    for text in tqdm(texts_in):
        new = gensim.utils.simple_preprocess(text, deacc=True)
        final.append(new)
    return final


def make_bigrams_trigrams(texts_in):
    bigram_phrases = gensim.models.Phrases(texts_in, min_count=5, threshold=100)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[texts_in], threshold=100)
    bigram = gensim.models.phrases.Phraser(bigram_phrases)
    trigram = gensim.models.phrases.Phraser(trigram_phrases)
    bigrams_out = [bigram[doc] for doc in texts_in]
    trigrams_out = [trigram[bigram[doc]] for doc in bigrams_out]
    return trigrams_out


def export_results(corpus, data, lda_model, texts):

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

    for i, topic_list in tqdm(enumerate(lda_model[corpus])):
        topic_list = sorted(topic_list, key=lambda x: (x[1]), reverse=True)

        try:
            main_keywords[i], main_topic[i], main_percentage[i] = find_topics(lda_model, topic_list[0])
        except IndexError:
            main_keywords[i], main_topic[i], main_percentage[i] = "None", "None", "None"

        try:
            second_keywords[i], second_topic[i], second_percentage[i] = find_topics(lda_model, topic_list[1])
        except IndexError:
            second_keywords[i], second_topic[i], second_percentage[i] = "None", "None", "None"

        try:
            third_keywords[i], third_topic[i], third_percentage[i] = find_topics(lda_model, topic_list[2])
        except IndexError:
            third_keywords[i], third_topic[i], third_percentage[i] = "None", "None", "None"

        text_snippets[i] = texts[i][:8]

    data['Main Topic'] = main_topic
    data['Main Topic Percentage'] = main_percentage
    data['Main Keywords'] = main_keywords
    data['Second Topic'] = second_topic
    data['Second Topic Percentage'] = second_percentage
    data['Second Keywords'] = second_keywords
    data['Third Topic'] = third_topic
    data['Third Topic Percentage'] = third_percentage
    data['Third Keywords'] = third_keywords
    data['Text Snippets'] = text_snippets

    return data


def reduce_corpus(corpus, id2word, tfidf):
    low_value = 0.03
    words = []
    words_missing_in_tfidf = []
    for i in tqdm(range(0, len(corpus))):
        bow = corpus[i]
        # low_value_words = [] #reinitialize to be safe. You can skip this.
        tfidf_ids = [i for i, value in tfidf[bow]]
        bow_ids = [i for i, value in bow]
        low_value_words = [i for i, value in tfidf[bow] if value < low_value]
        drops = low_value_words + words_missing_in_tfidf
        for item in drops:
            words.append(id2word[item])
        words_missing_in_tfidf = [i for i in bow_ids if
                                  i not in tfidf_ids]
        # The words with tf-idf score 0 will be missing

        new_bow = [b for b in bow if b[0] not in drops]
        corpus[i] = new_bow

    return corpus


def find_topics(ldamodel, topic):
    wp = ldamodel.show_topic(topic[0])
    keywords = ", ".join([word for word, prop in wp[:5]])
    topic_num = int(topic[0])
    topic_percentage = round(topic[1], 4)
    return keywords, topic_num, topic_percentage


if __name__ == "__main__":
    args = read_data.get_args()

    dataframe = pd.read_csv(args.load_dataframe, sep=";", on_bad_lines='warn', encoding_errors="ignore")
    print(f'{datetime.datetime.now()}: beginning LDA topic model')
    dataframe = gensim_lda(dataframe, args.lda_numtopics)
    print(datetime.datetime.now(), ": finished LDA")

    # export results
    dataframe.to_csv(args.output_document_path + '_gensim_res.csv', sep=';', index=False)

