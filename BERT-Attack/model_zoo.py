from transformers import DistilBertForMaskedLM, ElectraForMaskedLM, MobileBertForMaskedLM, RobertaForMaskedLM, BertForMaskedLM
from transformers import DistilBertTokenizer, ElectraTokenizer, MobileBertTokenizer, RobertaTokenizer, BertTokenizer
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, ElectraForSequenceClassification, MobileBertForSequenceClassification, RobertaForSequenceClassification


maskedLMs = {'bert':(BertTokenizer, BertForMaskedLM, "bert-base-uncased"),'distilbert':(DistilBertTokenizer, DistilBertForMaskedLM, "distilbert-base-uncased"), 'electra':(ElectraTokenizer, ElectraForMaskedLM, "google/electra-small-generator"), "mobilebert":(MobileBertTokenizer, MobileBertForMaskedLM, "google/mobilebert-uncased"), "roberta":(RobertaTokenizer, RobertaForMaskedLM, 'roberta-base')}

seqClassifiers = {'bert':(BertTokenizer, BertForSequenceClassification, "textattack/bert-base-uncased-yelp-polarity"), 'distilbert':(DistilBertTokenizer, DistilBertForSequenceClassification, "distilbert-base-uncased"), 'electra':(ElectraTokenizer, ElectraForSequenceClassification, "bhadresh-savani/electra-base-emotion"), "mobilebert":(MobileBertTokenizer, MobileBertForSequenceClassification, "lordtt13/emo-mobilebert"), "roberta":(RobertaTokenizer, RobertaForSequenceClassification, 'cardiffnlp/twitter-roberta-base-emotion')}