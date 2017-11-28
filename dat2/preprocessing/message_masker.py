import re
import spacy
from textacy.preprocess import (replace_phone_numbers, replace_numbers,
                                replace_emails, replace_urls,
                                normalize_whitespace)

XX_PATTERN = re.compile(r"x{2,}")
PROD_PATT = re.compile(r"(join\.?me)|(boldchat)", re.I)
NPS_USER_PATTERN = re.compile(r"[0-9]{2}-[0-9]{2}-.+sUser[0-9]+")

nlp = spacy.en.English()


def replace_entities(text):
    doc = nlp(text)
    charlist = list(text)
    offset = 0
    for tok in doc:
        if PROD_PATT.match(tok.text) is not None:
            ent_type = "PRODUCT"
            ent_iob = "B"
        elif tok.like_url:
            ent_type = "URL"
            ent_iob = "B"
        else:
            ent_type = tok.ent_type_
            ent_iob = tok.ent_iob_

        if ent_type in ["GPE", "LOC"]:
            ent_type = "LOCATION"
        elif ent_type == "NORP":
            ent_type = "GROUP_OF_PEOPLE"
        elif ent_type == "ORG":
            ent_type = "ORGANIZATION"
        elif ent_type == "DATE" and tok.text.isalpha():
            ent_type = None

        if ent_type:
            idx = tok.idx + offset
            tok_len = len(tok.text)
            replacement = "*{}*".format(ent_type) if ent_iob == "B" else ""
            replacement_len = len(replacement)
            charlist[idx:idx + tok_len] = list(replacement)
            offset += replacement_len - tok_len
    return "".join(charlist)


def replace_xx(text):
    return XX_PATTERN.sub("*CARDINAL*", text)


def replace_nps_user(text):
    return NPS_USER_PATTERN.sub("*PERSON*", text)


def normalize_message(text,
                      preprocess=(
                              replace_entities, replace_phone_numbers,
                              replace_numbers, replace_xx,
                              replace_emails, replace_urls,
                              normalize_whitespace
                      )):
    for fun in preprocess:
        text = fun(text)
    return text
