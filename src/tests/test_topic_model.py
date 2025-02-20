import unittest
import pandas as pd

from topic-rnews import topic_model

class TopicModelTest(unittest.TestCase):
    def test_leet_topic_model(self):
        df = pd.read_csv('test_df.csv', sep=';')
        df = topic_model.run_leet_topic(df)
        self.assertEqual()
