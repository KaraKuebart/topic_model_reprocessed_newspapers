import datetime
import pandas as pd
from read_data import get_args
from bertopic import BERTopic
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP

from sentence_transformers import SentenceTransformer

from topic_model_src import make_wordcloud
from preprocessing import drop_short_lines

if __name__ == "__main__":
    print(datetime.datetime.now(), ': beginning')
    # get args for import
    args = get_args()

    # import dataframe
    print(datetime.datetime.now(), ': import dataframe')
    news_df = pd.read_csv(args.load_dataframe, sep=';')

    print(datetime.datetime.now(), ': raw dataframe loaded. dropping short lines')
    news_df = drop_short_lines(news_df)

    print(datetime.datetime.now(), ': short lines dropped. generating embeddings')
    embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    embedding_model.encode(news_df["text"].astype(str), show_progress_bar=True)

    print(datetime.datetime.now(), ': define parameters')
    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)
    #  NOTICE adapt the nr_topics parameter as needed
    topic_model = BERTopic(embedding_model=embedding_model, nr_topics=1140, umap_model=umap_model, hdbscan_model=hdbscan_model)
    news_df['text'] = news_df['text'].astype(str)
    docs = news_df['text'].tolist()

    print(datetime.datetime.now(), ': beginning topic modeling')
    topics, probs = topic_model.fit_transform(docs)
    print(datetime.datetime.now(),
          f': finished BERTopic model. Results are: TOPIC LENGTH: {len(topics)}, PROBS LENGTH: {len(probs)}, DOCS LENGTH: {len(docs)}')
    print(datetime.datetime.now(), ': saving model')
    topic_model.save(args.output_document_path + '.pkl', serialization='pickle')

    print(datetime.datetime.now(), ': exporting to dataframes (one for information on topics found, the other is the updated dataset with topic distribution)')
    topic_ids = [i for i in range(-1, len(topic_model.topic_sizes_))]

    df = pd.DataFrame({'topic_ids': topic_ids})
    df['topic_sizes'] = topic_model.topic_sizes_
    df['topic mapper'] = topic_model.topic_mapper_
    df['topic_representations'] = topic_model.topic_representations_
    df['representative_docs'] = topic_model.representative_docs_
    df.to_csv(args.output_document_path + '_bertopic_topics.csv', sep=';', index=False)

    news_df['BERTopic'] = pd.Series(topics)
    news_df['BERTopic_prob'] = pd.Series(probs)
    news_df.to_csv(args.output_document_path + '_bertopic_res.csv', sep=';', index=False)

    print(datetime.datetime.now(), ': dataframes saved. Generating wordclouds')
    # print(datetime.datetime.now(), topic_model.topic_representations_)
    for topic_id in range(-1, len(topic_model.topic_representations_) - 1):
        topic = topic_model.topic_representations_[topic_id]
        topic_dict = {}
        for word in topic:
            topic_dict[word[0]] = word[1]
        make_wordcloud('BERTopic', topic_id, topic_dict)
