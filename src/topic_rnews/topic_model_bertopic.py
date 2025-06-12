print('code started running')
import datetime
print(datetime.datetime.now(), 'datetime imported. Importing pandas:')
import pandas as pd
print(datetime.datetime.now(), 'pandas imported. Importing read_data:')
from read_data import get_args
print(datetime.datetime.now(), 'read_data imported. Importing bertopic:')
from bertopic import BERTopic
print(datetime.datetime.now(), 'bertopic imported. Importing cuml.cluster HDBSCAN:')
from cuml.cluster import HDBSCAN
print(datetime.datetime.now(), 'HDBSCAN imported. Importing cuml.manifold UMAP:')
from cuml.manifold import UMAP

from sentence_transformers import SentenceTransformer

from topic_model_src import make_wordcloud

if __name__ == "__main__":
    print(datetime.datetime.now(), ': beginning')
    # get args for import
    args = get_args()

    # import dataframe
    print(datetime.datetime.now(), ': import dataframe')
    news_df = pd.read_csv(args.load_dataframe, sep=';')

    embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    # embeddings = embedding_model.encode(news_df["text"].astype(str), show_progress_bar=True)
    print(datetime.datetime.now(), ': define parameters')
    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)
    topic_model = BERTopic(embedding_model = embedding_model, umap_model=umap_model, hdbscan_model=hdbscan_model)
    news_df['text'] = news_df['text'].astype(str)
    docs = news_df['text'].tolist()

    print(datetime.datetime.now(), ': beginning topic modeling')
    topics, probs = topic_model.fit_transform(docs)
    print(datetime.datetime.now(), ': finished topic modeling')
    print(datetime.datetime.now(),
          f': finished BERTopic model. Results are: TOPIC LENGTH: {len(topics)}, PROBS LENGTH: {len(probs)}, DOCS LENGTH: {len(docs)}\n TOPICS: ',
          topics, '\n PROBS: ', probs, '\n\n\n')
    print(datetime.datetime.now(), ': Model contents are: \n TOPICS: ', topic_model.topics_, '\n PROBABILITIES: ',
          topic_model.probabilities_, '\n TOPIC SIZES: ', topic_model.topic_sizes_, '\n TOPIC MAPPER: ',
          topic_model.topic_mapper_, '\n TOPIC_REPRESENTATIONS: ', topic_model.topic_representations, '\n C_TF_IDF: ',
          topic_model.c_tf_idf_, '\n TOPIC LABELS: ', topic_model.topic_labels_, '\n TOPIC_EMBEDDINGS: ',
          topic_model.topic_embeddings_, '\n REPRESENTATIVE DOCS: ', topic_model.representative_docs_)
    doc_ids = [index for index in range(len(docs))]
    df = pd.DataFrame({ 'topic': topic_model.topics_, 'topic_labels' : topic_model.topic_labels_, 'topic_representations': topic_model.topic_representations_, 'docs': docs , 'topics as put out' : topics})
    df.to_csv(args.output_document_path + 'bertopic_test.csv', sep=';', index=False)
    news_df['BERTopic'] = pd.Series(topics)
    news_df['BERTopic_prob'] = pd.Series(probs)
    news_df.to_csv(args.output_document_path + '_bertopic_res.csv', sep=';', index=False)

    # for topic_nr in topic_model.topics_:

