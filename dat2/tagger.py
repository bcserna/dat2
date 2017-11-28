import numpy as np
import logging
import pandas as pd
import boto3
import botocore
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from dat2.feature_extraction.encoder import Encoder


class Tagger:
    S3_BUCKET = 'cesml'
    S3_MODEL_PATH = 'models/dialog_act_tagging/'
    MODEL_DIR = './models/'

    def __init__(self, classifier, encoder=None):
        self.classifier = classifier
        if encoder is None:
            self.encoder = Encoder()
        else:
            self.encoder = encoder

    def fit(self, X, y):
        self.classifier.fit(X, y)

    def fit_by_messages(self, messages, labels):
        X = self.encoder.transform(Tagger.messages_to_df(messages))
        self.classifier.fit(X, labels)

    def predict(self, X):
        return self.classifier.predict(X)

    def predict_by_messages(self, messages):
        X = self.encoder.transform(Tagger.messages_to_df(messages))
        return self.classifier.predict(X)

    def save(self, path=MODEL_DIR + 'tagger.pkl'):
        joblib.dump(self, path)

    @staticmethod
    def messages_to_df(messages):
        return pd.DataFrame(data=messages, columns=['MessageText'])

    @staticmethod
    def download_model_from_s3(s3_bucket=S3_BUCKET, s3_model_path=S3_MODEL_PATH,
                               download_destination=MODEL_DIR + 'tagger.pkl'):
        s3 = boto3.resource('s3')
        try:
            s3.Bucket(s3_bucket).download_file(s3_model_path,
                                               download_destination)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                logging.error('The object does not exist in s3.')
            else:
                raise

    @staticmethod
    def load_model(path=MODEL_DIR + 'tagger.pkl'):
        print('Loading model...')
        return joblib.load(filename=path)

    @staticmethod
    def load_data(path):
        return joblib.load(filename=path)


class EnsembleClassifier:
    def __init__(self, classifiers, voting: str, stacking_clf=LinearSVC()):
        self.classifiers = classifiers
        self.voting = voting
        self.stacking_clf = stacking_clf
        self.voting_clf = OneVsRestClassifier(stacking_clf)

    def fit(self, X, y):
        for clf in self.classifiers:
            clf.fit(X, y)
        if self.voting == 'stacking':
            preds = [clf.predict(X) for clf in self.classifiers]
            preds = np.concatenate(preds, axis=1)
            self.voting_clf.fit(preds, y)

    def predict(self, X):
        individual_preds = [clf.predict(X) for clf in self.classifiers]
        if self.voting == 'hard':
            preds = self.hard_vote(individual_preds)
        if self.voting == 'stacking':
            preds = self.voting_clf.predict(np.concatenate(individual_preds, axis=1))

        return preds

    @staticmethod
    def hard_vote(individual_preds):
        pred_sum = np.zeros(shape=individual_preds[0].shape)
        for preds in individual_preds:
            np.add(preds, pred_sum, out=pred_sum)

        nb_clf = len(individual_preds)
        threshold = 0.5
        voted_preds = [[0 if val / nb_clf < threshold else 1 for val in row]
                       for row in pred_sum]
        return np.array(voted_preds)


