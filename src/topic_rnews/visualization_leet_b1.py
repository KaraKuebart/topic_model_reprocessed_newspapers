import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle


def count_topic_frequency(topic_names:pd.Series) -> dict:
    return topic_names.value_counts().to_dict()

if __name__ == "__main__":
    import_fd = open("output/leet_resultsbatch_1.csv", encoding="utf-8", errors="ignore")
    batch_1 = pd.read_csv(import_fd, sep=";")

    batch_1.sort_values(['path', 'region'], inplace=True)
    batch_1.reset_index(drop=True, inplace=True)

    #ax = batch_1.plot.scatter(x='x', y='y')
    #plt.show()

    plt.figure(figsize=(25, 15))
    scatter = plt.scatter(batch_1['x'], batch_1['y'], c=batch_1['leet_labels'], cmap='tab10', edgecolors='k', alpha=0.75)

    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Topic Number', fontsize=30)

    # Labels and Title
    plt.xlabel('X Coordinate', fontsize=30)
    plt.ylabel('Y Coordinate', fontsize=30)
    plt.title('Topic Visualization', fontsize=30)
    plt.show()


    topic_dict = count_topic_frequency(batch_1['leet_labels'])
    print(topic_dict)
    most_important_topics = dict(list(topic_dict.items())[1:100])
    print(most_important_topics)
    for key in most_important_topics.keys():
        temp_df = batch_1.loc[batch_1['leet_labels'] == key]
        temp_df.to_csv(f"output/leet_topic_{key}.csv", sep=';', index=False)
        textlist = temp_df['text'].tolist()
        textstring = '\n'.join(textlist)
        textstring.replace('\n', ' ')
        wordlist = textstring.split(' ')
        word_frequencies = {}
        word_set = set(wordlist)
        for item in word_set:
            word_frequencies[item] = wordlist.count(item)

        with open(f'output/leet_words_topic_{key}.pkl', 'wb') as file:
            pickle.dump(word_frequencies, file)
        cloud = WordCloud(width=1000, height=600, background_color='white').generate(textstring)
        plt.figure(figsize=(10, 6))
        plt.imshow(cloud, interpolation='bilinear')
        plt.savefig(f'output/leet_words_topic_{key}.png')
        plt.close()

    most_important_topics_list = list(most_important_topics.keys())
    reduced_df = batch_1.loc[batch_1['leet_labels'].isin(most_important_topics_list)]

    plt.figure(figsize=(25, 15))
    scatter = plt.scatter(reduced_df['x'], reduced_df['y'], c=reduced_df['leet_labels'], cmap='tab10', edgecolors='k', alpha=0.75)

    # Colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Topic Number', fontsize=30)

    # Labels and Title
    plt.xlabel('X Coordinate', fontsize=30)
    plt.ylabel('Y Coordinate', fontsize=30)
    plt.title('Topic Visualization', fontsize=30)
    plt.show()

    print('breakpoint')