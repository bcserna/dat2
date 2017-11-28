MODEL_NAME = dialog_act_tagger_svm.pkl
EXTRACTOR_NAME = feature_extractor.pkl
MODEL_PREFIX = ''

.PHONY: all model upload_model preprocessed_chats turk_chats unlabeled_chats clean clean_data clean_model

all: ;

model:
	aws s3 cp \
		s3://cesml/models/dialog_act_tagging/$(MODEL_PREFIX)$(MODEL_NAME) \
		models/$(MODEL_PREFIX)$(MODEL_NAME)
	aws s3 cp \
		s3://cesml/models/dialog_act_tagging/$(MODEL_PREFIX)$(EXTRACTOR_NAME) \
		models/$(MODEL_PREFIX)$(EXTRACTOR_NAME)

upload_model:
	aws s3 cp \
		models/$(MODEL_PREFIX)$(MODEL_NAME) \
		s3://cesml/models/dialog_act_tagging/$(MODEL_PREFIX)$(MODEL_NAME)
	aws s3 cp \
		models/$(MODEL_PREFIX)$(EXTRACTOR_NAME) \
		s3://cesml/models/dialog_act_tagging/$(MODEL_PREFIX)$(EXTRACTOR_NAME)

preprocessed_chats:
	aws s3 cp \
		s3://cesml/datasets/joinme/dialogacts/results/processed/preprocessed_turk.csv \
		data/preprocessed_turk.csv

turk_chats:
	aws s3 cp \
		s3://cesml/datasets/joinme/dialogacts/results/processed/merged_messages_0_1500_normalized.csv \
		data/turk.csv

unlabeled_chats:
	aws s3 cp \
		s3://cesml/datasets/joinme/dialogacts/interim/chats.zip \
		data/chats.zip

clean: clean_data clean_model

clean_data:
	- rm -f data/*.*

clean_model:
	- rm -f *.pkl
