# pip install -U spacy[cuda121]
# pip install bertopic[spacy]

import spacy
import bertopic
import pandas as pd
from read_data import get_args


if __name__ == "__main__":
    # setup spacy
    spacy_nlp = spacy.load("de_core_news_sm")
    spacy.prefer_gpu()

    # get args for import
    args = get_args()

    # import dataframe
    news_df = pd.read_csv(args.load_dataframe, sep=',') # TODO remember to change this to ; !
    topic_model = bertopic.BERTopic()
    docs = news_df['text'].tolist()
    topics, probs = topic_model.fit(docs)
    df = pd.DataFrame({ 'topic': topic_model.topics_,'document': docs['id']})
    news_df['bertopic'] = pd.Series(topics)
    news_df['bertopic_prob'] = pd.Series(probs)
    news_df.to_csv(args.load_dataframe + '_bertopic_res.csv', index=False)
