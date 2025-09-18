import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    # ENTER HERE: NAME OF THE FILE AND THE THRESHOLD FROM WHICH YOU CONSINDER A DOCUMENT TO BE SUFFICIENTLY RELEVANT TO THE TOPIC TO BE INCLUDED IN YOUR SUB-DATASET
    # if you choose a threshold below 0.33, you may activate topic_df_2 and add it to the concatenation. Same goes for thresholds below 0.25 and topic_df_3.
    INPUT_FILE = 'all_tomoto_res.csv'
    MIN_RELEVANCE = 0.35

    out_file_base = INPUT_FILE.split('.', maxsplit=1)[0]
    dataset = pd.read_csv(f'output/{INPUT_FILE}', sep=';')
    topics = list(set(dataset['0_topic_nr']))
    topics_average_document_lengths = pd.DataFrame(columns=['topic_nr', 'nr_of_documents', 'average_document_length'])
    topics_average_document_lengths.set_index('topic_nr', inplace=True)
    for topic in tqdm(topics):
        try:
            inttopic = int(topic)
        except ValueError:
            continue
        topic_df_0 = dataset[(dataset['0_topic_nr'] == topic) & (dataset['0_topic_probability'] > MIN_RELEVANCE)]
        topic_df_1 = dataset[(dataset['1_topic_nr'] == topic) & (dataset['1_topic_probability'] > MIN_RELEVANCE)]
        # topic_df_2 = dataset[(dataset['2_topic_nr'] == topic) & (dataset['2_topic_probability'] > MIN_RELEVANCE)]
        # topic_df_3 = dataset[(dataset['3_topic_nr'] == topic) & (dataset['3_topic_probability'] > MIN_RELEVANCE)]
        topic_df = pd.concat([topic_df_0, topic_df_1])
        if len(topic_df) > 0:
            topic_df.to_csv(f'output/{out_file_base}_{inttopic}.csv', sep=';')
            topics_average_document_lengths.at[inttopic, 'average_document_length'] = round(topic_df['text'].str.len().mean())
            topics_average_document_lengths.at[inttopic, 'nr_of_documents'] = len(topic_df.index)
        else:
            topics_average_document_lengths.at[inttopic, 'average_document_length'] = 'NaN'
            topics_average_document_lengths.at[inttopic, 'nr_of_documents'] = 0
    topics_average_document_lengths.to_csv(f'output/{out_file_base}_average_document_lengths.csv', sep=';')
