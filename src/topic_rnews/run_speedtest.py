import read_data
import preprocessing
from tqdm import tqdm
from parallel_pandas import ParallelPandas
import datetime
import time
import pandas as pd
from resources.pythonic_resources import stopwords, consolidations, lemmata

if __name__ == "__main__":
    # initialize parallel pandas
    ParallelPandas.initialize(n_cpu=20, split_factor=4)
    # import data from numpy arrays
    print(datetime.datetime.now(), 'beginning')
    args = read_data.get_args()
    news_df = read_data.create_dataframe(args)
    print(datetime.datetime.now(), f'dataframe created (size:{len(news_df.index)}), joining headings and paragraphs:')

    # add headings and corresponding paragraphs together (if confidence is high).
    indices = news_df.index
    for i in tqdm(indices[:-2]):
        if news_df.loc[i]['class'] == 'heading' and news_df.loc[i+1]['class'] == 'paragraph' and float(news_df.loc[i]['confidence']) > 0.5 and float(news_df.loc[i+1]['confidence']) > 0.5 and news_df.loc[i]['region'] == int(news_df.loc[i+1]['region']) - 1:
            news_df.loc[i]['class'] = 'joined'
            news_df.loc[i]['confidence'] = (news_df.loc[i]['confidence'] + news_df.loc[i+1]['confidence']) / 2.0
            news_df.loc[i]['text'] = str(news_df.loc[i]['text']) + ' ' + str(news_df.loc[i+1]['text'])
            news_df.drop(news_df.loc[i+1], inplace=True)
    print(datetime.datetime.now(), 'headings and paragraphs joined')



    print(datetime.datetime.now(), ': dropping short lines. Length before_reduction:', len(news_df))
    news_df['text'] = news_df['text'].astype(str)
    news_df = news_df.drop(news_df[news_df['text'].str.len() < 50].index)
    print(datetime.datetime.now(), ': Length after reduction:', len(news_df))



    news_df_A = news_df.copy()
    print(datetime.datetime.now(), ': starting option A: Dictionaries in parallel processes')
    A_start = time.time()
    news_df_A['text'] = news_df_A['text'].str.lower()
    news_df_A = news_df_A.p_replace(to_replace="-\n", value="", regex=True)
    news_df_A = news_df_A.p_replace(to_replace="\n", value=" ", regex=True)


    # print(datetime.datetime.now(), ': applying consolidations')

    news_df_A = news_df_A.p_replace(consolidations, regex=True)


    # for i in tqdm(consolidations.index):
    #    dataset = dataset.p_replace(to_replace=consolidations.loc[i, "letters"],
                                  #value=consolidations.loc[i, "replace"],
                                  #regex=True)

    # print(datetime.datetime.now(), ': applying lemmata')
    news_df_A = news_df_A.p_replace(lemmata, regex=True)

    #for j in tqdm(lemmata.index):
    #    dataset = dataset.p_replace(to_replace=f""" {lemmata.loc[j].at["word"]} """,
                                  #value=f""" {lemmata.loc[j].at["replace"]} """, regex=True)

    # print(datetime.datetime.now(), ': applying stopwords')
    news_df_A = news_df_A.p_replace(stopwords, regex=True)
    #for k in tqdm(stopwords):
    #    dataset = dataset.p_replace(to_replace=f" {k} ", value=" ", regex=True)

    A_end = time.time()
    print(datetime.datetime.now(), ': elapsed time:', A_end - A_start)



    news_df_B = news_df.copy()
    print(datetime.datetime.now(), ': starting option B: Dictionaries in normal pandas')
    B_start = time.time()
    news_df_B['text'] = news_df_B['text'].str.lower()
    news_df_B = news_df_B.replace(to_replace="-\n", value="", regex=True)
    news_df_B = news_df_B.replace(to_replace="\n", value=" ", regex=True)


    #print(datetime.datetime.now(), ': applying consolidations')

    news_df_B = news_df_B.replace(consolidations, regex=True)


    # for i in tqdm(consolidations.index):
    #    dataset = dataset.p_replace(to_replace=consolidations.loc[i, "letters"],
                                  #value=consolidations.loc[i, "replace"],
                                  #regex=True)

    #print(datetime.datetime.now(), ': applying lemmata')
    news_df_B = news_df_B.replace(lemmata, regex=True)

    #for j in tqdm(lemmata.index):
    #    dataset = dataset.p_replace(to_replace=f""" {lemmata.loc[j].at["word"]} """,
                                  #value=f""" {lemmata.loc[j].at["replace"]} """, regex=True)

    #print(datetime.datetime.now(), ': applying stopwords')
    news_df_B = news_df_B.replace(stopwords, regex=True)
    #for k in tqdm(stopwords):
    #    dataset = dataset.p_replace(to_replace=f" {k} ", value=" ", regex=True)

    B_end = time.time()
    print(datetime.datetime.now(), ': elapsed time:', B_end - B_start)



    consolidations = pd.read_csv('resources/consolidations.csv', sep=';')
    lemmata = pd.read_csv('resources/lemmata.csv', sep=';')
    stopwords = ['aber', 'alle', 'allem', 'allen', 'aller', 'alles', 'als', 'also', 'am', 'an', 'ander', 'andere',
                     'anderem', 'anderen', 'anderer', 'anderes', 'anderm', 'andern', 'anderr', 'anders', 'auch', 'auf',
                     'aus', 'bei', 'bin', 'bis', 'bist', 'da', 'damit', 'dann', 'der', 'den', 'des', 'dem', 'die',
                     'das', 'dass', 'daß', 'derselbe', 'derselben', 'denselben', 'desselben', 'demselben', 'dieselbe',
                     'dieselben', 'dasselbe', 'dazu', 'dein', 'deine', 'deinem', 'deinen', 'deiner', 'deines', 'denn',
                     'derer', 'dessen', 'dich', 'dir', 'du', 'dies', 'diese', 'diesem', 'diesen', 'dieser', 'dieses',
                     'doch', 'dort', 'durch', 'ein', 'eine', 'einem', 'einen', 'einer', 'eines', 'einig', 'einige',
                     'einigem', 'einigen', 'einiger', 'einiges', 'einmal', 'er', 'ihn', 'ihm', 'es', 'etwas', 'euer',
                     'eure', 'eurem', 'euren', 'eurer', 'eures', 'für', 'gegen', 'gewesen', 'hab', 'habe', 'haben',
                     'hat', 'hatte', 'hatten', 'hier', 'hin', 'hinter', 'ich', 'mich', 'mir', 'ihr', 'ihre', 'ihrem',
                     'ihren', 'ihrer', 'ihres', 'euch', 'im', 'in', 'indem', 'ins', 'ist', 'jede', 'jedem', 'jeden',
                     'jeder', 'jedes', 'jene', 'jenem', 'jenen', 'jener', 'jenes', 'jetzt', 'kann', 'kein', 'keine',
                     'keinem', 'keinen', 'keiner', 'keines', 'können', 'könnte', 'machen', 'man', 'manche', 'manchem',
                     'manchen', 'mancher', 'manches', 'mein', 'meine', 'meinem', 'meinen', 'meiner', 'meines', 'mit',
                     'muss', 'musste', 'nach', 'nicht', 'nichts', 'noch', 'nun', 'nur', 'ob', 'oder', 'ohne', 'sehr',
                     'sein', 'seine', 'seinem', 'seinen', 'seiner', 'seines', 'selbst', 'sich', 'sie', 'ihnen', 'sind',
                     'so', 'solche', 'solchem', 'solchen', 'solcher', 'solches', 'soll', 'sollte', 'sondern', 'sonst',
                     'über', 'um', 'und', 'uns', 'unsere', 'unserem', 'unseren', 'unser', 'unseres', 'unter', 'viel',
                     'vom', 'von', 'vor', 'während', 'war', 'waren', 'warst', 'was', 'weg', 'weil', 'weiter', 'welche',
                     'welchem', 'welchen', 'welcher', 'welches', 'wenn', 'werde', 'werden', 'wie', 'wieder', 'will',
                     'wir', 'wird', 'wirst', 'wo', 'wollen', 'wollte', 'würde', 'würden', 'zu', 'zum', 'zur', 'zwar',
                     'zwischen', 'ab', 'abgerufen', 'abgerufene', 'abgerufener', 'abgerufenes', 'acht', 'allein',
                     'allerdings', 'allerlei', 'allgemein', 'allmählich', 'allzu', 'alsbald', 'andererseits',
                     'andernfalls', 'anerkannt', 'anerkannte', 'anerkannter', 'anerkanntes', 'anfangen', 'anfing',
                     'angefangen', 'angesetze', 'angesetzt', 'angesetzten', 'angesetzter', 'ansetzen', 'anstatt',
                     'arbeiten', 'aufgehört', 'aufgrund', 'aufhören', 'aufhörte', 'aufzusuchen', 'ausdrücken',
                     'ausdrückt', 'ausdrückte', 'ausgenommen', 'ausser', 'ausserdem', 'author', 'autor', 'außen',
                     'außer', 'außerdem', 'außerhalb', 'bald', 'bearbeite', 'bearbeiten', 'bearbeitete', 'bearbeiteten',
                     'bedarf', 'bedurfte', 'bedürfen', 'befragen', 'befragte', 'befragten', 'befragter', 'begann',
                     'beginnen', 'begonnen', 'behalten', 'behielt', 'beide', 'beiden', 'beiderlei', 'beides', 'beim',
                     'beinahe', 'beitragen', 'beitrugen', 'bekannt', 'bekannte', 'bekannter', 'bekennen', 'benutzt',
                     'bereits', 'berichten', 'berichtet', 'berichtete', 'berichteten', 'besonders', 'besser',
                     'bestehen', 'besteht', 'beträchtlich', 'bevor', 'bezüglich', 'bietet', 'bisher', 'bislang',
                     'bleiben', 'blieb', 'bloss', 'bloß', 'brachte', 'brachten', 'brauchen', 'braucht', 'bringen',
                     'bräuchte', 'bsp.', 'bzw', 'böden', 'ca.', 'dabei', 'dadurch', 'dafür', 'dagegen', 'daher',
                     'dahin', 'damals', 'danach', 'daneben', 'dank', 'danke', 'danken', 'dannen', 'daran', 'darauf',
                     'daraus', 'darf', 'darfst', 'darin', 'darum', 'darunter', 'darüber', 'darüberhinaus', 'davon',
                     'davor', 'demnach', 'denen', 'dennoch', 'derart', 'derartig', 'derem', 'deren', 'derjenige',
                     'derjenigen', 'derzeit', 'deshalb', 'desto', 'deswegen', 'diejenige', 'diesseits', 'dinge',
                     'direkt', 'direkte', 'direkten', 'direkter', 'doppelt', 'dorther', 'dorthin', 'drauf', 'drei',
                     'dreißig', 'drin', 'dritte', 'drunter', 'drüber', 'dunklen', 'durchaus', 'durfte', 'durften',
                     'dürfen', 'dürfte', 'eben', 'ebenfalls', 'ebenso', 'ehe', 'eher', 'eigenen', 'eigenes',
                     'eigentlich', 'einbaün', 'einerseits', 'einfach', 'einführen', 'einführte', 'einführten',
                     'eingesetzt', 'einigermaßen', 'eins', 'einseitig', 'einseitige', 'einseitigen', 'einseitiger',
                     'einst', 'einstmals', 'einzig', 'ende', 'entsprechend', 'entweder', 'ergänze', 'ergänzen',
                     'ergänzte', 'ergänzten', 'erhalten', 'erhielt', 'erhielten', 'erhält', 'erneut', 'erst', 'erste',
                     'ersten', 'erster', 'eröffne', 'eröffnen', 'eröffnet', 'eröffnete', 'eröffnetes', 'etc', 'etliche',
                     'etwa', 'fall', 'falls', 'fand', 'fast', 'ferner', 'finden', 'findest', 'findet', 'folgende',
                     'folgenden', 'folgender', 'folgendes', 'folglich', 'fordern', 'fordert', 'forderte', 'forderten',
                     'fortsetzen', 'fortsetzt', 'fortsetzte', 'fortsetzten', 'fragte', 'frau', 'frei', 'freie',
                     'freier', 'freies', 'fuer', 'fünf', 'gab', 'ganz', 'ganze', 'ganzem', 'ganzen', 'ganzer', 'ganzes',
                     'gar', 'gbr', 'geb', 'geben', 'geblieben', 'gebracht', 'gedurft', 'geehrt', 'geehrte', 'geehrten',
                     'geehrter', 'gefallen', 'gefiel', 'gefälligst', 'gefällt', 'gegeben', 'gehabt', 'gehen', 'geht',
                     'gekommen', 'gekonnt', 'gemacht', 'gemocht', 'gemäss', 'genommen', 'genug', 'gern', 'gesagt',
                     'gesehen', 'gestern', 'gestrige', 'getan', 'geteilt', 'geteilte', 'getragen', 'gewissermaßen',
                     'gewollt', 'geworden', 'ggf', 'gib', 'gibt', 'gleich', 'gleichwohl', 'gleichzeitig',
                     'glücklicherweise', 'gmbh', 'gratulieren', 'gratuliert', 'gratulierte', 'gute', 'guten', 'gängig',
                     'gängige', 'gängigen', 'gängiger', 'gängiges', 'gänzlich', 'haette', 'halb', 'hallo', 'hast',
                     'hattest', 'hattet', 'heraus', 'herein', 'heute', 'heutige', 'hiermit', 'hiesige', 'hinein',
                     'hinten', 'hinterher', 'hoch', 'hundert', 'hätt', 'hätte', 'hätten', 'höchstens', 'igitt', 'immer',
                     'immerhin', 'important', 'indessen', 'info', 'infolge', 'innen', 'innerhalb', 'insofern',
                     'inzwischen', 'irgend', 'irgendeine', 'irgendwas', 'irgendwen', 'irgendwer', 'irgendwie',
                     'irgendwo', 'ja', 'je', 'jedenfalls', 'jederlei', 'jedoch', 'jemand', 'jenseits', 'jährig',
                     'jährige', 'jährigen', 'jähriges', 'kam', 'kannst', 'kaum', 'keinerlei', 'keineswegs', 'klar',
                     'klare', 'klaren', 'klares', 'klein', 'kleinen', 'kleiner', 'kleines', 'koennen', 'koennt',
                     'koennte', 'koennten', 'komme', 'kommen', 'kommt', 'konkret', 'konkrete', 'konkreten', 'konkreter',
                     'konkretes', 'konnte', 'konnten', 'könn', 'könnt', 'könnten', 'künftig', 'lag', 'lagen', 'langsam',
                     'lassen', 'laut', 'lediglich', 'leer', 'legen', 'legte', 'legten', 'leicht', 'leider', 'lesen',
                     'letze', 'letzten', 'letztendlich', 'letztens', 'letztes', 'letztlich', 'lichten', 'liegt',
                     'liest', 'links', 'längst', 'längstens', 'mache', 'machst', 'macht', 'machte', 'machten', 'mag',
                     'magst', 'mal', 'mancherorts', 'manchmal', 'mann', 'margin', 'mehr', 'mehrere', 'meist', 'meiste',
                     'meisten', 'meta', 'mindestens', 'mithin', 'mochte', 'morgen', 'morgige', 'muessen', 'muesst',
                     'muesste', 'musst', 'mussten', 'muß', 'mußt', 'möchte', 'möchten', 'möchtest', 'mögen', 'möglich',
                     'mögliche', 'möglichen', 'möglicher', 'möglicherweise', 'müssen', 'müsste', 'müssten', 'müßt',
                     'müßte', 'nachdem', 'nacher', 'nachhinein', 'nacht', 'nahm', 'natürlich', 'neben', 'nebenan',
                     'nehmen', 'nein', 'neu', 'neue', 'neuem', 'neuen', 'neuer', 'neues', 'neun', 'nie', 'niemals',
                     'niemand', 'nimm', 'nimmer', 'nimmt', 'nirgends', 'nirgendwo', 'nutzen', 'nutzt', 'nutzung',
                     'nächste', 'nämlich', 'nötigenfalls', 'nützt', 'oben', 'oberhalb', 'obgleich', 'obschon', 'obwohl',
                     'oft', 'per', 'pfui', 'plötzlich', 'pro', 'reagiere', 'reagieren', 'reagiert', 'reagierte',
                     'rechts', 'regelmäßig', 'rief', 'rund', 'sage', 'sagen', 'sagt', 'sagte', 'sagten', 'sagtest',
                     'sang', 'sangen', 'schlechter', 'schließlich', 'schnell', 'schon', 'schreibe', 'schreiben',
                     'schreibens', 'schreiber', 'schwierig', 'schätzen', 'schätzt', 'schätzte', 'schätzten', 'sechs',
                     'sect', 'sehe', 'sehen', 'sehrwohl', 'seht', 'sei', 'seid', 'seit', 'seitdem', 'seite', 'seiten',
                     'seither', 'selber', 'senke', 'senken', 'senkt', 'senkte', 'senkten', 'setzen', 'setzt', 'setzte',
                     'setzten', 'sicher', 'sicherlich', 'sieben', 'siebte', 'siehe', 'sieht', 'singen', 'singt',
                     'sobald', 'sodaß', 'soeben', 'sofern', 'sofort', 'sog', 'sogar', 'solange', 'solc hen', 'solch',
                     'sollen', 'sollst', 'sollt', 'sollten', 'solltest', 'somit', 'sonstwo', 'sooft', 'soviel',
                     'soweit', 'sowie', 'sowohl', 'spielen', 'später', 'startet', 'startete', 'starteten', 'statt',
                     'stattdessen', 'steht', 'steige', 'steigen', 'steigt', 'stets', 'stieg', 'stiegen', 'such',
                     'suchen', 'sämtliche', 'tages', 'tat', 'tatsächlich', 'tatsächlichen', 'tatsächlicher',
                     'tatsächliches', 'tausend', 'teile', 'teilen', 'teilte', 'teilten', 'titel', 'total', 'trage',
                     'tragen', 'trotzdem', 'trug', 'trägt', 'tun', 'tust', 'tut', 'txt', 'tät', 'ueber', 'umso',
                     'unbedingt', 'ungefähr', 'unmöglich', 'unmögliche', 'unmöglichen', 'unmöglicher', 'unnötig',
                     'unse', 'unsem', 'unsen', 'unserer', 'unserm', 'unses', 'unten', 'unterbrach', 'unterbrechen',
                     'unterhalb', 'unwichtig', 'usw', 'vergangen', 'vergangene', 'vergangener', 'vergangenes', 'vermag',
                     'vermutlich', 'vermögen', 'verrate', 'verraten', 'verriet', 'verrieten', 'version', 'versorge',
                     'versorgen', 'versorgt', 'versorgte', 'versorgten', 'versorgtes', 'veröffentlichen',
                     'veröffentlicher', 'veröffentlicht', 'veröffentlichte', 'veröffentlichten', 'veröffentlichtes',
                     'viele', 'vielen', 'vieler', 'vieles', 'vielleicht', 'vielmals', 'vier', 'vollständig', 'voran',
                     'vorbei', 'vorgestern', 'vorher', 'vorne', 'vorüber', 'völlig', 'wachen', 'waere', 'wann', 'warum',
                     'weder', 'wegen', 'weitere', 'weiterem', 'weiteren', 'weiterer', 'weiteres', 'weiterhin', 'weiß',
                     'wem', 'wen', 'wenig', 'wenige', 'weniger', 'wenigstens', 'wenngleich', 'wer', 'werdet', 'weshalb',
                     'wessen', 'wichtig', 'wieso', 'wieviel', 'wiewohl', 'willst', 'wirklich', 'wodurch', 'wogegen',
                     'woher', 'wohin', 'wohingegen', 'wohl', 'wohlweislich', 'wolle', 'wollt', 'wollten', 'wolltest',
                     'wolltet', 'womit', 'woraufhin', 'woraus', 'worin', 'wurde', 'wurden', 'währenddessen', 'wär',
                     'wäre', 'wären', 'z.B.', 'zahlreich', 'zehn', 'zeitweise', 'ziehen', 'zieht', 'zog', 'zogen',
                     'zudem', 'zuerst', 'zufolge', 'zugleich', 'zuletzt', 'zumal', 'zurück', 'zusammen', 'zuviel',
                     'zwanzig', 'zwei', 'zwölf', 'ähnlich', 'übel', 'überall', 'überallhin', 'überdies', 'übermorgen',
                     'übrig', 'übrigens', '.', '&', '%', '$', '"""', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                     'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ä', 'ö', 'ü',
                     'ss', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '#', '+', '~', '^', '°', '-']


    print(datetime.datetime.now(), ': starting option C: for-loop with parallelization on operations')
    C_start = time.time()
    news_df['text'] = news_df['text'].str.lower()
    news_df = news_df.p_replace(to_replace="-\n", value="", regex=True)
    news_df = news_df.p_replace(to_replace="\n", value=" ", regex=True)


    #print(datetime.datetime.now(), ': applying consolidations')

    #news_df = news_df.p_replace(consolidations, regex=True)

    for i in tqdm(consolidations.index):
        news_df = news_df.p_replace(to_replace=consolidations.loc[i, "letters"],
                                  value=consolidations.loc[i, "replace"],
                                  regex=True)

    #print(datetime.datetime.now(), ': applying lemmata')
    #news_df = news_df.p_replace(lemmata, regex=True)

    for j in tqdm(lemmata.index):
        news_df = news_df.p_replace(to_replace=f""" {lemmata.loc[j].at["word"]} """,
                                  value=f""" {lemmata.loc[j].at["replace"]} """, regex=True)

    #print(datetime.datetime.now(), ': applying stopwords')
    #news_df = news_df.p_replace(stopwords, regex=True)
    for k in tqdm(stopwords):
        news_df = news_df.p_replace(to_replace=f" {k} ", value=" ", regex=True)

    C_end = time.time()
    print(datetime.datetime.now(), ': elapsed time:', C_end - C_start)