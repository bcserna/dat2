import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
from scipy import sparse
import warnings

from dat2.feature_extraction.feature_extractors import *
from dat2.feature_extraction.feature_functions import *
from dat2.util import LABELS


class Encoder:
    available_features = ['structural', 'wordvec', 'word_tfidf', 'wh_questions', 'greets', 'emotes', 'question_mark',
                          'first_n_tfidf', 'md_window', 'infersent', 'char_tfidf']

    def __init__(self, use_features):
        # self.scaler = StandardScaler()
        self.scaler = MinMaxScaler()

        featurizers = []

        if 'structural' in set(use_features):
            structural = FunctionFeaturizer(
                number_of_words,
                sender_info,
                previous_msg_labels,
                previous_same_author_msg_labels,
            )
            featurizers.append(('structural', structural))

        if 'wordvec' in use_features:
            wordvec = FunctionFeaturizer(avg_wordvec)
            featurizers.append(('wordvec', wordvec))

        if 'word_tfidf' in use_features:
            word_tfidf = ChatTfidf(use_idf=True, ngram_range=(1, 2))
            featurizers.append(('word_tfidf', word_tfidf))

        if 'wh_questions' in use_features:
            wh_questions = WordExistence({'who', 'where', 'why', 'when', 'how', 'what', 'which', 'whose', 'whom'})
            featurizers.append(('wh_questions', wh_questions))

        if 'greets' in use_features:
            greets = WordExistence({'bye', 'goodbye', 'hello', 'hi'})
            featurizers.append(('greets', greets))

        if 'emotes' in use_features:
            emotes = WordExistence({
                ':)', ':-)', ':(', ':-(',
                ':d', ':-d', 'xd',
                ';d', ';-d', ';)', ';-)', ';(', ';-(',
                ':o', ':s', ':\\', ':/', ':|',
                'lol', 'lmao', 'rofl'
            })
            featurizers.append(('emotes', emotes))

        if 'question_mark' in use_features:
            question_mark = WordExistence({'?'})
            featurizers.app(('question_mark', question_mark))

        if 'first_n_tfidf' in use_features:
            first_n_tfidf = FirstNWordsTfidf(n=4, use_idf=True, ngram_range=(1, 2))
            featurizers.append(('first_n_tfidf', first_n_tfidf))

        if 'md_window' in use_features:
            md_window = MdWindowTfIdf(use_idf=False, ngram_range=(1, 4), words_before=2, words_after=2)
            featurizers.append(('md_window', md_window))

        if 'infersent' in use_features:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                infersent = InfersentEncoder()
                featurizers.append(('infersent', infersent))

        if 'char_tfidf' in use_features:
            char_tfidf = ChatTfidf(use_idf=False, ngram_range=(3, 6), analyzer='char', min_df=5)
            featurizers.append(('char_tfidf', char_tfidf))

        self.feature_unifier = FeatureUnion(featurizers, n_jobs=1)

        self.normalizer = MaxAbsScaler()
        self.kbest = SelectKBest(chi2, k=3000)
        self.pca = TruncatedSVD(n_components=1200)
        self.etc = SelectFromModel(ExtraTreesClassifier(n_jobs=10))

    def fit(self, messages: pd.DataFrame, y=None):
        train_vecs = self.feature_unifier.fit_transform(messages, y)
        if sparse.issparse(train_vecs):
            train_vecs = train_vecs.todense()
        self.scaler.fit(train_vecs)

    def transform(self, messages: pd.DataFrame):
        feature_vectors = self.feature_unifier.transform(messages)
        if sparse.issparse(feature_vectors):
            feature_vectors = feature_vectors.todense()
        feature_vectors = self.scaler.transform(feature_vectors)
        # print('Feature vectors\' shape: ', feature_vectors.shape)
        return feature_vectors

    @staticmethod
    def extract_labels(messages: pd.DataFrame):
        labels = messages[LABELS].values
        return labels
