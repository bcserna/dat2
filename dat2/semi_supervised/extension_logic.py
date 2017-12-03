import pandas as pd
import numpy as np
from cld2 import detect as cld2_detect
from sklearn import clone
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from tqdm import tqdm
from dat2.preprocessing import preprocessor
from dat2.util import chats_msg_list, messages_to_df, LABELS


def is_english(text):
    try:
        is_reliable, _, best_guesses = cld2_detect(text, bestEffort=True)
        if is_reliable is False:
            pass
        lang = best_guesses[0][1]
        return lang == "en"
    except Exception as e:
        return False


def english_ratio(chat):
    return np.average(chat.Text.apply(is_english).values)


def message_count(chat):
    return chat.shape[0]


def customer_message_count(chat):
    return chat[chat.User == 'customer'].shape[0]


def wordcount(chat):
    return np.sum([len(str(msg).split()) for msg in chat.Text.values])


def filter_chats(chats, labeled):
    try:
        return [
            chat for chat in tqdm(chats, desc='Filtering chats')
            if chat.ChatID[0] not in set(labeled.ChatId.values)
            and 7 <= message_count(chat) <= 21
            and customer_message_count(chat) >= 3
            and wordcount(chat) > 70
            and english_ratio(chat) > 0.95
        ]
    except:
        pass


class Extender:
    def __init__(self, unlabeled_messages, labeled_messages, unlabeled_vectors, labeled_vectors, labeled_gold_standard,
                 classifier: OneVsRestClassifier):
        # ids = unlabeled_messages.MessageId
        # self.unlabeled = [(id_, text, vec) for id_, text, vec
        #                   in zip(ids, messages, unlabeled_vecs)]
        self.unlabeled_messages = np.array(unlabeled_messages.MessageText)
        self.unlabeled_vectors = unlabeled_vectors
        self.labeled_messages = labeled_messages
        self.labeled_vectors = labeled_vectors
        self.labeled_gold = labeled_gold_standard
        self.classifier = classifier
        self.extension_messages = {l: [] for l in LABELS}
        self.extension_vectors = {l: [] for l in LABELS}
        self.extension_labels = {l: [] for l in LABELS}
        self.scores = []

    def self_learning(self, batch_size=1000, nb_iter=5, threshold=0.9):
        self.extension_messages = {l: [] for l in LABELS}
        self.extension_vectors = {l: [] for l in LABELS}
        self.extension_labels = {l: [] for l in LABELS}
        indices = np.arange(len(self.unlabeled_messages))
        np.random.shuffle(indices)
        for _ in tqdm(range(nb_iter), desc='Batch'):
            batch_indices = indices[:batch_size]
            batch_vectors = self.unlabeled_vectors[batch_indices]
            batch_messages = self.unlabeled_messages[batch_indices]
            pred_proba = self.classifier.predict_proba(batch_vectors)

            # Extension
            for pred, message, vector, i in zip(pred_proba, batch_messages, batch_vectors, batch_indices):
                for l, p in zip(LABELS, pred):
                    if p > threshold:
                        self.extension_messages[l].append(message)
                        self.extension_vectors[l].append(vector)
                        self.extension_labels[l].append(1)
                    # elif p < 1 - threshold:
                    #     self.extension_vectors[l].append(vector)
                    #     self.extension_labels[l].append(0)

            self.retrain_individual_classifiers(self.classifier, self.labeled_vectors, self.labeled_gold)

            indices = indices[batch_size:]
            np.append(arr=indices, values=batch_indices)

    def cross_val_predict(self, cv=5):
        kf = KFold(n_splits=cv)
        clf = clone(self.classifier)
        clf.fit(X=self.labeled_vectors, y=self.labeled_gold)
        predictions = None
        for train_index, test_index in kf.split(self.labeled_vectors):
            labeled_train_x = self.labeled_vectors[train_index]
            labeled_train_y = self.labeled_gold[train_index]
            labeled_test_x = self.labeled_vectors[test_index]
            self.retrain_individual_classifiers(clf, labeled_train_x, labeled_train_y)
            fold_pred = clf.predict(X=labeled_test_x)
            if predictions is None:
                predictions = fold_pred
            else:
                predictions = np.concatenate((predictions, fold_pred))

        return predictions

    def retrain_individual_classifiers(self, classifier, labeled_x, labeled_y):
        for l, estimator, i in zip(LABELS, classifier.estimators_, range(len(LABELS))):
            if len(self.extension_labels[l]) > 0:
                extended_data = np.concatenate((labeled_x, self.extension_vectors[l]))
                extended_labels = np.concatenate((labeled_y[:, i], self.extension_labels[l]))
            else:
                extended_data = labeled_x
                extended_labels = labeled_y[:, i]
            estimator.fit(X=extended_data, y=extended_labels)
