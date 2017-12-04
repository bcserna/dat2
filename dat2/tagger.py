import numpy as np
import logging
import pandas as pd
import boto3
import botocore
from sklearn.base import BaseEstimator
from sklearn.externals import joblib
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_predict
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
    def __init__(self, classifiers, voting: str, voting_clf=LinearSVC()):
        assert voting in ['hard', 'soft', 'hard-stacking', 'soft-stacking']
        self.classifiers = classifiers
        self.voting = voting
        self.voting_clf = OneVsRestClassifier(voting_clf, n_jobs=-1)

    def fit(self, X, y):
        for clf_name in self.classifiers:
            self.classifiers[clf_name].fit(X[clf_name], y)
        if self.voting == 'hard-stacking':
            preds = [self.classifiers[clf_name].predict(X[clf_name]) for clf_name in self.classifiers]
            preds = np.concatenate(preds, axis=1)
            self.voting_clf.fit(preds, y)
        if self.voting == 'soft-stacking':
            proba_preds = [self.classifiers[clf_name].predict_proba(X[clf_name]) for clf_name in self.classifiers]
            proba_preds = np.concatenate(proba_preds, axis=1)
            self.voting_clf.fit(proba_preds, y)

    def cross_val_predict(self, X, y):
        if self.voting == 'hard':
            individual_preds = [cross_val_predict(estimator=self.classifiers[clf_name],
                                                  X=X[clf_name],
                                                  y=y,
                                                  cv=5)
                                for clf_name in self.classifiers]
            preds = self.vote_average(individual_preds)
        if self.voting == 'soft':
            for c in self.classifiers:
                self.classifiers[c] = ProbabilityPredictionWrapper(self.classifiers[c])
            individual_proba_preds = [cross_val_predict(estimator=self.classifiers[clf_name],
                                                        X=X[clf_name],
                                                        y=y,
                                                        cv=5)
                                      for clf_name in self.classifiers]
            preds = self.vote_average(individual_proba_preds)
        if self.voting == 'hard-stacking':
            individual_preds = [cross_val_predict(estimator=self.classifiers[clf_name],
                                                  X=X[clf_name],
                                                  y=y,
                                                  cv=5)
                                for clf_name in self.classifiers]
            preds = cross_val_predict(estimator=self.voting_clf, X=np.concatenate(individual_preds, axis=1), y=y, cv=5)
        if self.voting == 'soft-stacking':
            for c in self.classifiers:
                self.classifiers[c] = ProbabilityPredictionWrapper(self.classifiers[c])
            individual_proba_preds = [cross_val_predict(estimator=self.classifiers[clf_name],
                                                        X=X[clf_name],
                                                        y=y,
                                                        cv=5)
                                      for clf_name in self.classifiers]
            preds = cross_val_predict(estimator=self.voting_clf, X=np.concatenate(individual_proba_preds, axis=1), y=y,
                                      cv=5)
        return preds

    def predict(self, X):
        if self.voting == 'hard':
            individual_preds = [self.classifiers[clf_name].predict(X[clf_name]) for clf_name in self.classifiers]
            preds = self.vote_average(individual_preds)
        if self.voting == 'soft':
            individual_proba_preds = [self.classifiers[clf_name].predict_proba(X[clf_name]) for clf_name in
                                      self.classifiers]
            preds = self.vote_average(individual_proba_preds)
        if self.voting == 'hard-stacking':
            individual_preds = [self.classifiers[clf_name].predict(X[clf_name]) for clf_name in self.classifiers]
            preds = self.voting_clf.predict(np.concatenate(individual_preds, axis=1))
        if self.voting == 'soft-stacking':
            individual_proba_preds = [self.classifiers[clf_name].predict_proba(X[clf_name]) for clf_name in
                                      self.classifiers]
            preds = self.voting_clf.predict(np.concatenate(individual_proba_preds, axis=1))
        return preds

    def predict_proba(self, X):
        if self.voting == 'hard':
            individual_preds = [self.classifiers[clf_name].predict(X[clf_name]) for clf_name in self.classifiers]
            pred_proba = np.average(individual_preds, axis=0)
        if self.voting == 'soft':
            individual_proba_preds = [self.classifiers[clf_name].predict_proba(X[clf_name]) for clf_name in
                                      self.classifiers]
            pred_proba = np.average(individual_proba_preds, axis=0)
        if self.voting == 'hard-stacking':
            individual_preds = [self.classifiers[clf_name].predict(X[clf_name]) for clf_name in self.classifiers]
            pred_proba = self.voting_clf.predict_proba(np.concatenate(individual_preds, axis=1))
        if self.voting == 'soft-stacking':
            individual_proba_preds = [self.classifiers[clf_name].predict_proba(X[clf_name]) for clf_name in
                                      self.classifiers]
            pred_proba = self.voting_clf.predict_proba(np.concatenate(individual_proba_preds, axis=1))
        return pred_proba

    @staticmethod
    def vote_average(individual_preds):
        pred_avg = np.average(individual_preds, axis=0)
        threshold = 0.5
        voted_preds = [[0 if val < threshold else 1 for val in row]
                       for row in pred_avg]
        return np.array(voted_preds)


class ProbabilityPredictionWrapper(BaseEstimator):
    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict_proba(X)
