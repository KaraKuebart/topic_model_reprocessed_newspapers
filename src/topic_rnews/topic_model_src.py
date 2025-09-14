import pickle

import matplotlib.pyplot as plt
from wordcloud import WordCloud


def make_wordcloud(method:str, topic_number:int, frequencies:dict, path:str)-> None:
    with open(f'{path}/{method}_words_topic_{topic_number}.pkl', 'wb') as file:
        pickle.dump(frequencies, file)
    cloud = WordCloud(width=1000, height=600, background_color='white').generate_from_frequencies(frequencies)
    plt.figure(figsize=(10, 6))
    plt.imshow(cloud, interpolation='bilinear')
    plt.title(f'Words in Topic {topic_number}', fontsize=30)
    plt.savefig(f'{path}/{method}_words_topic_{topic_number}.png')
    plt.close()
