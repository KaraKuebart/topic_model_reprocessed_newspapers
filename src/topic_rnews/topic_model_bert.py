import pandas as pd
from read_data import get_args
from bertopic import BERTopic
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from sentence_transformers import SentenceTransformer


if __name__ == "__main__":
    # get args for import
    args = get_args()

    # import dataframe
    news_df = pd.read_csv(args.load_dataframe, sep=';')

    embedding_model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    embeddings = embedding_model.encode(news_df["text"], show_progress_bar=True)

    umap_model = UMAP(n_components=5, n_neighbors=15, min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)
    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model)
    news_df['text'] = news_df['text'].astype(str)
    docs = news_df['text'].tolist()
    topics, probs = topic_model.fit_transform(docs)
    df = pd.DataFrame({ 'topic': topic_model.topics_,'document': docs['id']})
    news_df['bertopic'] = pd.Series(topics)
    news_df['bertopic_prob'] = pd.Series(probs)
    filename = args.load_dataframe.split('.')[0]
    news_df.to_csv(filename + '_bertopic_res.csv', index=False)
