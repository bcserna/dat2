import io
import logging
from zipfile import ZipFile
import pandas as pd
import numpy as np
import re
import spacy
from tqdm import tqdm
from sklearn.externals import joblib

pattern = re.compile(r'\*\w+\*')
spacy.en.English.Defaults.token_match = pattern.match
nlp = spacy.en.English(token_match=pattern.match)

PREV = 'Prev-'
SAME_AUTH = 'SameAuth-'
IS_STARTER = 'IsStarter'

LABELS = np.array([
    'Answer-Ack', 'Answer-No', 'Answer-Yes', 'Greeting-Closing',
    'Greeting-Opening', 'Question-Open', 'Question-YesNo',
    'Request-Help', 'Request-Info', 'Request-Other', 'Socialact-Apology',
    'Socialact-Downplayer', 'Socialact-Gratitude', 'Statement-Complaint',
    'Statement-ExpressiveNegative', 'Statement-ExpressivePositive',
    'Statement-Informative', 'Statement-OfferHelp', 'Statement-Promise',
    'Statement-SuggestAction'
])


def messages_to_df(messages):
    return pd.DataFrame(data=messages, columns=['MessageText'])


def chat_msg_list(chat: pd.DataFrame):
    return [chat.loc[i:i, :] for i in range(chat.shape[0])]


def chats_msg_list(chats):
    return [msg for chat in chats for msg in chat_msg_list(chat)]


def chats_msg_dataframe(chats):
    return pd.concat(chats_msg_list(chats)).reset_index(drop=True)


def label_vectors_to_string(label_vectors):
    comp = np.zeros(len(LABELS))
    return [LABELS[v > comp].tolist() for v in label_vectors]


def messages_to_chats(messages: pd.DataFrame):
    return [messages[messages.ChatId == chat_id].reset_index(drop=True) for chat_id in messages.ChatId.unique()]


def confusion_matrix(prediction_vectors, gold_vectors):
    conf_matrix = pd.DataFrame(index=LABELS, columns=LABELS).fillna(0)

    pred_labels = label_vectors_to_string(prediction_vectors)
    gold_labels = label_vectors_to_string(gold_vectors)

    for pred, gold in zip(pred_labels, gold_labels):
        pred, gold = set(pred), set(gold)
        true_positive = pred & gold
        false_positive = pred - gold
        false_negative = gold - pred

        for label in true_positive:
            conf_matrix.loc[label, label] += 1

        for label_false_positive in false_positive:
            for label_false_negative in false_negative:
                conf_matrix.loc[label_false_positive, label_false_negative] += 1

    return conf_matrix


def _read_chats_from_zip(chats_zip, n_files=None):
    with ZipFile(chats_zip) as zf:
        for name in tqdm(zf.namelist()[:n_files], 'Reading chats'):
            if name.endswith(".csv"):
                try:
                    with zf.open(name) as f:
                        df = pd.read_csv(io.TextIOWrapper(f), delimiter="\t")
                        yield df
                except Exception as e:
                    logging.error("Could not read {}. {}".format(name, e))


def read_chats_from_zip(chats_zip, n_files=None):
    return list(_read_chats_from_zip(chats_zip, n_files=n_files))


def read_preprocessed_chats(preprocessed_path):
    return messages_to_chats(pd.read_csv(preprocessed_path))


def print_messages_with_labels(messages, predicted_labels, gold_labels, difference_only=False):
    for pred, gold, i in zip(predicted_labels, gold_labels, range(len(messages))):
        pred = set(pred)
        gold = set(gold)

        if difference_only:
            pred = pred - gold
            gold = gold - pred

        print(messages[i])
        print('    Predicted:     ', sorted(pred))
        print('    Gold standard: ', sorted(gold), '\n')


def save_vecs(turk_vecs, prem_vecs, name):
    joblib.dump(turk_vecs, ''.join(['./data/turk_', name, '.pkl']))
    joblib.dump(prem_vecs, ''.join(['./data/prem_', name, '.pkl']))
