
Dialog Act Tagger
-----------------
A multi-label dialog act tagger for chat messages.

* Training a model:

.. code-block:: python

  >>> from dat2.tagger import Tagger
  >>> from sklearn.multiclass import OneVsRestClassifier
  >>> from sklearn.svm import LinearSVC
  >>> classifier = OneVsRestClassifier(LinearSVC())
  >>> tagger = Tagger(classifier)
  >>> tagger.fit_to_messages(messages, labels, fit_encoder=True)

* Using a pre-trained model:

.. code-block:: python

  >>> from dat2.tagger import Tagger
  >>> from sklearn.externals import joblib
  >>> tagger = joblib.load('./models/tagger.pkl')
  >>> messages = ([
  ... 'How can I help you?',
  ... 'Thank you, have a nice day!',
  ... 'What is the difference between the free and pro versions?'
  ])
  >>> predictions = tagger.tag(messages)
  >>> predictions
  [['Statement-OfferHelp'],
  ['Greeting-Closing', 'Socialact-Gratitude'],
  ['Question-Open', 'Request-Info']]

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

* Important note:

To use the Infersent embeddings in the encoder, you will need to download a pre-trained model and word vectors (visit https://github.com/facebookresearch/InferSent for more).
