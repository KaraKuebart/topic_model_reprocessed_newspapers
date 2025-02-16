import pandas as pd
from leet_topic import leet_topic

import read_data

df = pd.read_csv('bonn_newspaper_1866_pre.csv', sep=';')

leet_df, topic_data = leet_topic.LeetTopic(df,
                                           document_field="text",
                                           html_filename="newspaper_leet_topic_0_9_0.html",
                                           extra_fields=["hdbscan_labels"],
                                           spacy_model="de_core_news_sm",
                                           max_distance=.5)

leet_df.to_csv('bonn_newspaper_1866_leet2.csv', sep=';')
