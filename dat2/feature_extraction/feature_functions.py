from dat2.util import PREV, SAME_AUTH, LABELS, nlp


def sender_info(messages):
    return messages.User.apply(lambda sender: [1] if sender == 'Operator' else [0]).tolist()


def number_of_words(messages):
    return messages.MessageText.apply(lambda text: [len(text.split())]).tolist()


def previous_msg_labels(messages):
    return messages[[PREV + l for l in LABELS]].values


def previous_same_author_msg_labels(messages):
    return messages[[PREV + SAME_AUTH + l for l in LABELS]].values


def avg_wordvec(messages):
    return messages.MessageText.apply(lambda text: nlp(text).vector).tolist()
