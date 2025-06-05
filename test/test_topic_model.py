import unittest
import pandas as pd

import src.topic_rnews.topic_model_leet
from src.topic_rnews import topic_model_src

class TopicModelTest(unittest.TestCase):
    def test_leet_topic_model(self):
        df = pd.read_csv('test/test_df.csv', sep=';')
        df = src.topic_rnews.topic_model_leet.run_leet_topic(df)
        self.assertEqual(df.columns.values.tolist(), ['path', 'region', 'class', 'confidence', 'text', 'x', 'y', 'hdbscan_labels', 'leet_labels'])
