import pandas as pd
import numpy as np
from os import path

from tqdm import tqdm

from dat2.preprocessing import message_masker
from dat2.util import LABELS, PREV, IS_STARTER, SAME_AUTH, nlp, read_chats_from_zip, messages_to_chats, \
    chats_msg_dataframe


def attach_previous_msg_labels(chat: pd.DataFrame):
    for l in LABELS:
        chat[PREV + l] = chat[l].shift(1)

    is_starter = np.arange(0, chat.shape[0], 1)
    chat[IS_STARTER] = pd.Series(is_starter)
    return chat.fillna(0)


def attach_previous_same_author_msg_labels(chat: pd.DataFrame):
    operator_msgs = chat[chat.User == 'Operator'].reset_index(drop=True)
    client_msgs = chat[chat.User == 'Client'].reset_index(drop=True)

    for l in LABELS:
        operator_msgs[PREV + SAME_AUTH + l] = operator_msgs[l].shift(1).fillna(0)
        client_msgs[PREV + SAME_AUTH + l] = client_msgs[l].shift(1).fillna(0)

    chat = pd.concat([operator_msgs, client_msgs])
    chat.sort_values(by='Created', inplace=True)

    return chat.reset_index(drop=True)


def attack_timestamps_to_chat(chat, chats):
    chat_id = chat.ChatId.values[0]
    for c in chats:
        if c.ChatID.values[0] == chat_id:
            return chat.join(c[['ChatMessageID', 'Created']].set_index('ChatMessageID'), on='MessageId')


def attach_timestamps(turk_chats, chats):
    return [attack_timestamps_to_chat(chat, chats).reset_index(drop=True)
            for chat in tqdm(messages_to_chats(turk_chats), 'Attaching timestamps')]


def filter_non_labeled(chat):
    chat = chat[chat[LABELS].apply(sum, axis=1) != 0]
    return chat.reset_index(drop=True)


def mask_chat(chat: pd.DataFrame):
    chat.MessageText = chat.MessageText.apply(message_masker.normalize_message)
    return chat


def preprocess_chat_pipeline(chat):
    chat = chat.drop(['Other', 'HITId'], axis=1, errors='ignore')
    chat = filter_non_labeled(chat)
    chat = mask_chat(chat)
    chat = attach_previous_msg_labels(chat)
    chat = attach_previous_same_author_msg_labels(chat)
    return chat


def preprocess(chats):
    all_chats_path = path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))), 'data/chats.zip')
    all_chats = read_chats_from_zip(all_chats_path)
    chats = attach_timestamps(chats, all_chats)
    chats = [preprocess_chat_pipeline(chat) for chat in chats]
    results = chats_msg_dataframe(chats)
    return results


def preprocess_messages_df(messages_df):
    messages_df = messages_df.drop(['Other', 'HITId'], axis=1, errors='ignore')
    messages_df = filter_non_labeled(messages_df)
    messages_df = mask_chat(messages_df)
    return messages_df
