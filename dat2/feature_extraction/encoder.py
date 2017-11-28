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
    def __init__(self):
        self.scaler = StandardScaler()
        # self.scaler = MinMaxScaler()

        structural = FunctionFeaturizer(
            number_of_words,
            # sender_info,
            # previous_msg_labels,
            # previous_same_author_msg_labels,
        )
        wordvec = FunctionFeaturizer(avg_wordvec)
        word_vectorizer = ChatTfidf(
            use_idf=True,
            ngram_range=(1, 2)
        )
        wh_questions = WordExistence({
            'who', 'where', 'why', 'when', 'how', 'what', 'which', 'whose', 'whom'
        })
        greets = WordExistence({
            'bye', 'goodbye', 'hello', 'hi'
        })
        emotes = WordExistence({
            ':)', ':-)', ':(', ':-(',
            ':d', ':-d', 'xd',
            ';d', ';-d', ';)', ';-)', ';(', ';-(',
            ':o', ':s', ':\\', ':/', ':|',
            'lol', 'lmao', 'rofl'
        })
        question_mark = WordExistence({
            '?'
        })
        first_n_words = FirstNWordsTfidf(
            n=4,
            use_idf=True,
            ngram_range=(1, 2)
        )
        md_vectorizer = MdWindowTfIdf(
            use_idf=False,
            ngram_range=(1, 4),
            words_before=2,
            words_after=2,
        )

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            infersent = InfersentEncoder()

        char_tfidf = ChatTfidf(
            use_idf=False,
            ngram_range=(3, 6),
            analyzer='char',
            min_df=5
        )

        self.feature_unifier = FeatureUnion([
            # ('char_tfidf', char_tfidf)
            # ('word_tfidf', word_vectorizer),
            # ('first_n_words', first_n_words),
            ('infersent', infersent),
            # ('wordvec', wordvec),
            # ('md_tfidf', md_vectorizer),
            # ('structural', structural),
            # ('wh_questions', wh_questions),
            # ('greets', greets),
            # ('emotes', emotes),
            # ('question_mark', question_mark),
        ],
            n_jobs=1
        )

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
        print('Feature vectors\' shape: ', feature_vectors.shape)
        return feature_vectors

    @staticmethod
    def extract_labels(messages: pd.DataFrame):
        labels = messages[LABELS].values
        return labels
