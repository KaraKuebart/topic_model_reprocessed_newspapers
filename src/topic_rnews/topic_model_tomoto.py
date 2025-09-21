import datetime
import sys
import numpy as np
import pandas as pd
import psutil
import tomotopy as tp
from tqdm import tqdm

from read_data import get_args
from topic_model_src import make_wordcloud


def tomoto_lda(dataframe: pd.DataFrame, num_topics:int=None, out_filename:str='tomotopy_lda'):
    num_docs = dataframe.shape[0]
    if num_topics is None:
        num_topics = round(np.power(num_docs, 5/12))
    mdl = tp.LDAModel(min_cf = int(np.power(num_docs, 1/3)), rm_top=int(np.log(num_docs)*20), k=num_topics, seed=42)
    print(datetime.datetime.now(), f': importing data into tomoto_lda model. RAM usage: {round(psutil.virtual_memory().used / 1e9)} GB')



    for i in tqdm(dataframe.index):
        text = str(dataframe.at[i, 'text']).split()
        mdl.add_doc(text)
    sys.stdout.flush()
    print(datetime.datetime.now(), f': import done. Starting training. RAM usage: {round(psutil.virtual_memory().used / 1e9)} GB')
    for j in range(0, 100, 10): # this is how bab2min recommends using his Topic Model. It will result 100 iterations, grouped in 10s.
        mdl.train(10)
    sys.stdout.flush()
    print(datetime.datetime.now(), f': training done. Saving model. RAM usage: {round(psutil.virtual_memory().used / 1e9)} GB')
    mdl.save('output/' + out_filename +'.bin')
    print(datetime.datetime.now(), f': removed top words', mdl.removed_top_words, '\n', '')

    print(datetime.datetime.now(), f': starting export to dataframe. RAM usage: {round(psutil.virtual_memory().used / 1e9)} GB')
    # trim spaces on text column, to avoid matching errors
    dataframe['text'] = dataframe['text'].str.strip()
    dataframe['text'] = dataframe['text'].str.replace('         ', ' ')
    dataframe['text'] = dataframe['text'].str.replace('     ', ' ')
    dataframe['text'] = dataframe['text'].str.replace('   ', ' ')
    dataframe['text'] = dataframe['text'].str.replace('  ', ' ')

    # Exporting tomotopy is tricky, since it does not reliably keep the document order. (This might have to do with word removal and empty docs).
    # The larger a dataset is, the more likely will there be indexing mismatches when entering results into the dataframe.
    # If we would not account for mismatches, this would 'move' topic distributions up whenever there is an
    # offset (= tomoto is missing a document that is in the dataframe, which would then get the topic distributions of the next document in the tomotopy model).
    # These mismatches would continue affecting all further documents from the first occuring offset.
    # To avoid such mismatches, the following code will check if the document text in the tomotopy model matches the one in the dataframe.
    # If documents do not match, this code will find the document (if it is in the dataframe at all), enter the results and keep track of offsets,
    # both for transparency and to improve efficiency.
    offset = 0
    error_counter = 0
    latest_overoffset_doc = None
    latest_overoffset = 0
    overoffset_row = 0
    for doc_nr in tqdm(range(len(mdl.docs))):
        try:
            doc_inst = mdl.docs[doc_nr]
            doc_text = str(doc_inst).split('"')[1]
            topic_tuplist = doc_inst.get_topics(top_n=4)

            # let's find the correct index in the DF to enter the results in.
            if dataframe.at[doc_nr + offset, 'text'] == doc_text:
                for m in range(0, 4):
                    dataframe.at[doc_nr + offset, f'{m}_topic_nr'] = topic_tuplist[m][0]
                    dataframe.at[doc_nr + offset, f'{m}_topic_probability'] = topic_tuplist[m][1]

            elif dataframe.at[doc_nr + offset + 1, 'text'] == doc_text:
                offset += 1
                print(datetime.datetime.now(), f'doc nr {doc_nr} has an offset. Offset increased to {offset}')
                for m in range(0, 4):
                    dataframe.at[doc_nr + offset, f'{m}_topic_nr'] = topic_tuplist[m][0]
                    dataframe.at[doc_nr + offset, f'{m}_topic_probability'] = topic_tuplist[m][1]

            elif doc_text in dataframe['text']:
                k = dataframe[dataframe['text'] == doc_text].index[0]
                current_overoffset = k - doc_nr
                for m in range(0, 4):
                    dataframe.at[k, f'{m}_topic_nr'] = topic_tuplist[m][0]
                    dataframe.at[k, f'{m}_topic_probability'] = topic_tuplist[m][1]
                print(datetime.datetime.now(),
                      f'Document nr {doc_nr} has been found in df at index row {k}. Difference: {current_overoffset}. Current offset: {offset}')
                if latest_overoffset_doc == doc_nr - 1 and latest_overoffset == current_overoffset:
                    overoffset_row += 1
                    if overoffset_row == 4:
                        offset = current_overoffset
                        print(datetime.datetime.now(), f'five subsequent over-offsets of {current_overoffset} found. Offset changed to {offset}')
                        overoffset_row = 0
                else:
                    overoffset_row = 0
                latest_overoffset = current_overoffset
                latest_overoffset_doc = doc_nr

            else:
                error_counter += 1
                print(datetime.datetime.now(),
                      f': Index error on tomotopy document {doc_nr}. No matching entry in Dataframe found. Total number of matching errors: {error_counter}. Offset: {offset}')
            # k = None

        except IndexError:
            error_counter += 1
            print(datetime.datetime.now(),
                  f': Index error on tomotopy document {doc_nr}. No matching entry in Dataframe found. Total number of matching errors: {error_counter}. Offset: {offset}')
        except Exception as e:
            error_counter += 1
            print(datetime.datetime.now(), f': Exception raised on tomotopy document {doc_nr}: {e}. Total number of matching errors: {error_counter}')
    sys.stdout.flush()
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
    # vocabulary = mdl.vocabs
    # words_importance_dict = {}
    # print(f'{datetime.datetime.now()}: creating word clouds for each topic. RAM usage: {round(psutil.virtual_memory().used / 1e9)} GB')
    # for topic_id in tqdm(range(num_topics)):
    #     topic_words = mdl.get_topic_word_dist(topic_id, normalize=True)
    #     for w in range(len(topic_words)):
    #         words_importance_dict[vocabulary[w]] = topic_words[w]
    #
    #     make_wordcloud('tomoto', topic_id, words_importance_dict, args.output_document_path)
