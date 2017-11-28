from os import path
import numpy as np
import torch
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from dat2.util import nlp


class WordExistence(TransformerMixin):
    def __init__(self, words: set):
        self.words = words

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        feature_vectors = [[1 if word in set(text.split()) else 0 for word in self.words]
                           for text in X.MessageText.values]
        return feature_vectors


class ChatTfidf(TransformerMixin):
    def __init__(self, use_idf: bool, tokenizer=str.split, ngram_range=(1, 1), min_df=5, analyzer='word'):
        self.vectorizer = TfidfVectorizer(
            use_idf=use_idf,
            tokenizer=tokenizer,
            ngram_range=ngram_range,
            min_df=min_df,
            analyzer=analyzer
        )

    def fit(self, X, y=None):
        messages = X.MessageText.apply(lemmatize_message).values
        self.vectorizer.fit(messages, y)
        return self

    def transform(self, X):
        messages = X.MessageText.apply(lemmatize_message).values
        return self.vectorizer.transform(messages)


class FirstNWordsTfidf(ChatTfidf):
    def __init__(self, n: int, use_idf: bool, ngram_range, min_df=5):
        super().__init__(use_idf=use_idf, ngram_range=ngram_range, min_df=min_df)
        self.n = n

    def fit(self, X, y=None):
        first_n_words_texts = [' '.join(text.split()[:self.n]) for text in
                               X.MessageText.apply(lemmatize_message).values]
        self.vectorizer.fit(first_n_words_texts, y)
        return self

    def transform(self, X):
        first_n_words = [' '.join([token.lemma_ for token in nlp(str(text))]) for text in
                         X.MessageText.apply(lemmatize_message).values]
        return self.vectorizer.transform(first_n_words)


class MdWindowTfIdf(ChatTfidf):
    def __init__(self, use_idf: bool, ngram_range=(1, 1), words_before=2, words_after=2):
        super().__init__(use_idf=use_idf, ngram_range=ngram_range)
        self.words_after = words_after
        self.words_before = words_before

    def md_windows(self, text_tags):
        return [text_tags[max(0, i - self.words_before):i + self.words_after + 1]
                for i, tag in enumerate(text_tags) if tag == 'MD']

    def windows_from_messages(self, messages):
        tags_list = [[token.tag_ for token in nlp(str(text))]
                     for text in messages]
        windows = [self.md_windows(tags) for tags in tags_list]
        return windows

    def fit(self, X, y=None):
        windows = [' '.join(window) for windows in self.windows_from_messages(X.MessageText.values)
                   for window in windows]
        self.vectorizer.fit(windows, y)
        return self

    def transform(self, X):
        feature_vectors = []
        for msg_windows in self.windows_from_messages(X.MessageText.values):
            msg_windows_as_text = []
            if msg_windows:
                msg_windows_as_text = [' '.join(window) for window in msg_windows]
            else:
                msg_windows_as_text.append('')
            msg_vectors = self.vectorizer.transform(msg_windows_as_text).toarray()
            feature_vectors.append(sum(msg_vectors))
        return feature_vectors


class FunctionFeaturizer(TransformerMixin):
    def __init__(self, *featurizers):
        self.featurizers = featurizers

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        features = [f(X) for f in self.featurizers]
        interleaved = [np.concatenate([feature[i] for feature in features])
                       for i in range(len(features[0]))]

        return interleaved


class InfersentEncoder(TransformerMixin):
    def __init__(self, vocab_k_words=100000):
        infersent_dir = path.join(path.dirname(path.abspath(__file__)), 'infersent')
        model_path = path.join(infersent_dir, 'infersent.allnli.pickle')
        glove_path = path.join(infersent_dir, 'glove.840B.300d.txt')
        self.infersent = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.infersent.set_glove_path(glove_path)
        self.infersent.build_vocab_k_words(K=vocab_k_words)

    def fit(self, X=None, y=None):
        return self

    def transform(self, X):
        texts = X.MessageText.values
        return self.infersent.encode(sentences=texts, tokenize=True)


def lemmatize_message(message: str):
    return ' '.join([token.lemma_ for token in nlp(message)])
