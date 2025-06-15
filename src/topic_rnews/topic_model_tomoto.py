import numpy as np
import pandas as pd
import tomotopy as tp

from read_data import get_args
from topic_model_src import make_wordcloud

import datetime
from tqdm import tqdm


def tomoto_lda(dataframe: pd.DataFrame, num_topics:int=None, out_filename:str='tomotopy_lda'):
    num_docs = dataframe.shape[0]
    if num_topics is None:
        num_topics = round(np.power(num_docs, 1/6))
    mdl = tp.LDAModel(min_cf = int(np.power(num_docs, 1/3)), rm_top=int(np.log(num_docs)*20), k=num_topics, seed=42)
    print(datetime.datetime.now(), ': importing data into tomoto_lda model')



    for i in tqdm(dataframe.index):
        text = str(dataframe.at[i, 'text']).split()
        mdl.add_doc(text)
    print(datetime.datetime.now(), ': import done. Starting training')
    for j in range(0, 100, 10): # this is how bab2min recommends using his Topic Model. It will result 100 iterations, grouped in 10s.
        mdl.train(10)
    print(datetime.datetime.now(), ': training done. Saving model')
    mdl.save('output/' + out_filename +'.bin')
    print(datetime.datetime.now(), ': removed top words', mdl.removed_top_words, '\n', '')

    print(datetime.datetime.now(), ': starting export to dataframe')
    for k in tqdm(dataframe.index):
        try:
            doc_inst= mdl.docs[k]
            topic_dists = doc_inst.get_topic_dist()
            topic_tuplist = []
            for l, prob in enumerate(topic_dists):
                topic_tuplist.append((l, round(prob, 4)))
            topic_tuplist.sort(reverse=True, key=lambda tup: tup[1])
            for m in range(0, 4):
                dataframe.at[k, f'{m}_topic_nr'] = topic_tuplist[m][0]
                dataframe.at[k, f'{m}_topic_probability'] = topic_tuplist[m][1]
        except IndexError:
            print(datetime.datetime.now(), f': Index error on document {k}. Number of Documents in the Dataframe and Number of Documents in the Model do not match')
        except Exception as e:
            print(datetime.datetime.now(), f': Exception raised on document {k}: {e}')
    return dataframe, mdl, num_topics


# TODO: implement inference (to train on a fraction of the data, then infer to the rest)


if __name__ == "__main__":
    # get args for import
    args = get_args()

    # import dataframe
    news_df = pd.read_csv(args.load_dataframe, sep=';')

    # to make sure the indices in the dataframe will match those in the model, we reset the dataframe index:
    news_df.reset_index(drop=True, inplace=True)

    # run tomoto LDAModel
    news_df, mdl, num_topics = tomoto_lda(news_df)
    # save results file
    news_df.to_csv(args.output_document_path + '_tomoto_res.csv', sep=';', index=False)
    vocabulary = mdl.vocabs
    words_importance_dict = {}
    print(f'{datetime.datetime.now()}: creating word clouds for each topic')
    for topic_id in tqdm(range(num_topics)):
        topic_words = mdl.get_topic_word_dist(topic_id, normalize=True)
        for w in range(len(topic_words)):
            words_importance_dict[vocabulary[w]] = topic_words[w]

        make_wordcloud('tomoto', topic_id, words_importance_dict)
