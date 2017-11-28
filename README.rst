
Dialog Act Tagger
-----------------
A multi-label dialog act tagger for chat messages.

* Downloading the training data:

.. code-block:: bash

  $ make preprocessed_chats

* Training a model:

.. code-block:: python

  >>> from dialog_act_tagger.tagger import Tagger
  >>> Tagger.create_model(model_path='...', extractor_path='...', model_prefix='...')    # default paths can be used

* Downloading a pre-trained model:

.. code-block:: python

  >>> from dialog_act_tagger.tagger import Tagger
  >>> Tagger.download_model_from_s3(s3_bucket='...', s3_model_path='...', model_prefix='...',
                                    model_name='...', extractor_name='...')

This will download *s3://s3_bucket/s3_model_path/model_prefix+model_name* and *s3://s3_bucket/s3_model_path/model_prefix+extractor_name*.

If you just want to download the default model, simply use:

.. code-block:: python

  >>> Tagger.download_model_from_s3()

* Downloading a pre-trained model with make:

.. code-block:: bash

  $ make model MODEL_NAME='dummy_tagger.pkl' EXTRACTOR_NAME='dummy_extractor.pkl' MODEL_PREFIX='20170822_'

This will download *20170822_dummy_tagger.pkl* and *20170822_dummy_extractor.pkl*.

If you just want to download the default model (*dialog_act_tagger_svm.pkl* and *feature_extractor.pkl*), simply use:

.. code-block:: bash

  $ make model

* Using a pre-trained model:

.. code-block:: python

  >>> from dialog_act_tagger.tagger import Tagger
  >>> tagger = Tagger(model_name='...', extractor_name='...', model_prefix='...')    # default paths are the same as for the Makefile
  >>> messages = ([
  ... 'How can I help you?',
  ... 'Thank you, have a nice day!',
  ... 'What is the difference between the free and pro versions?'
  ])
  >>> predictions = tagger.predict(messages, output_type='str')
  >>> predictions
  [['Statement-OfferHelp'],
  ['Greeting-Closing', 'Socialact-Gratitude'],
  ['Question-Open', 'Request-Info']]

* Evaluating a model:

.. code-block:: python

  >>> predictions = tagger.cross_val_predict()
  >>> report = Tagger.classification_report(predictions)

* Printing messages with the adherent predicted and gold standard labels:

.. code-block:: python

  >>> from dialog_act_tagger.util import print_messages_with_labels
  >>> gold_labels = [['Statement-OfferHelp'], ['Greeting-Closing', 'Socialact-Gratitude'], ['Question-Open', 'Request-Info']]
  >>> print_messages_with_labels(messages, predictions, gold_labels, difference_only=False)
  How can I help you?
      Predicted:      ['Statement-OfferHelp']
      Gold standard:  ['Statement-OfferHelp']
  Thank you, have a nice day!
      Predicted:      ['Greeting-Closing', 'Socialact-Gratitude']
      Gold standard:  ['Greeting-Closing', 'Socialact-Gratitude']
  What is the difference between the free and pro versions?
      Predicted:      ['Question-Open']
      Gold standard:  ['Question-Open', 'Request-Info']

* Uploading a model:

.. code-block:: bash

  $ make upload_model MODEL_NAME='...' EXTRACTOR_NAME='...' MODEL_PREFIX='...'

Note: changing only the *MODEL_PREFIX* parameter and leaving the other two with their default values should be enough in most cases.
