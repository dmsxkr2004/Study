import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd

from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
path = "D:/_data/dacon/open/"
datasets = pd.read_csv(path+"train_data.csv", header = 0, index_col = 0)
test_sets  = pd.read_csv(path+"test_data.csv", header = 0, index_col = 0)
sub_sets   = pd.read_csv(path+"sample_submission.csv", header = 0, index_col = 0)

datasets = datasets.dropna(how='any') # Null 값이 존재하는 행 제거
datasets = datasets.reset_index(drop=True)
print(datasets.isnull().values.any()) # Null 값이 존재하는지 확인

test_sets = test_sets.dropna(how='any') # Null 값이 존재하는 행 제거
test_sets = test_sets.reset_index(drop=True)
print(test_sets.isnull().values.any()) # Null 값이 존재하는지 확인

print("premise 최대 길이:", datasets['premise'].map(len).max()) # premise 최대 길이: 90
print("hypothesis 최대 길이:", datasets['hypothesis'].map(len).max()) # hypothesis 최대 길이: 103

print("premise 최대 길이:", test_sets['premise'].map(len).max()) # premise 최대 길이: 90
print("hypothesis 최대 길이:", test_sets['hypothesis'].map(len).max()) # hypothesis 최대 길이: 75

max_seq_len = 100

valid = datasets[20000:]
datasets = datasets[:20000]

from transformers import AutoTokenizer

model_name = "klue/roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

from tqdm import tqdm
import numpy as np

def convert_examples_to_features(sent_list1, sent_list2, max_seq_len, tokenizer):

    input_ids, attention_masks, token_type_ids = [], [], []

    for sent1, sent2 in tqdm(zip(sent_list1, sent_list2), total=len(sent_list1)):
        encoding_result = tokenizer.encode_plus(sent1, sent2, 
                                                max_length=max_seq_len, 
                                                pad_to_max_length=True)

        input_ids.append(encoding_result['input_ids'])
        attention_masks.append(encoding_result['attention_mask'])
        token_type_ids.append(encoding_result['token_type_ids'])

    input_ids = np.array(input_ids, dtype=int)
    attention_masks = np.array(attention_masks, dtype=int)
    token_type_ids = np.array(token_type_ids, dtype=int)

    return (input_ids, attention_masks, token_type_ids)

X_train = convert_examples_to_features(datasets['premise'], datasets['hypothesis'], 
                                       max_seq_len=max_seq_len, 
                                       tokenizer=tokenizer)

X_valid = convert_examples_to_features(valid['premise'], valid['hypothesis'], 
                                       max_seq_len=max_seq_len, 
                                       tokenizer=tokenizer)

X_test = convert_examples_to_features(test_sets['premise'], test_sets['hypothesis'], 
                                       max_seq_len=max_seq_len, 
                                       tokenizer=tokenizer)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(datasets['label'])
y_valid = le.transform(valid['label'])

label_idx = dict(zip(list(le.classes_), le.transform(list(le.classes_))))
print(label_idx)
'''
{'contradiction': 0, 'entailment': 1, 'neutral': 2}
'''
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import TruncatedNormal
from transformers import TFAutoModel

class TFBertForSequenceClassification(Model):
    def __init__(self, model_name):
        super(TFBertForSequenceClassification, self).__init__()
        self.bert = TFAutoModel.from_pretrained(model_name, 
                                                num_labels=3, 
                                                from_pt=True)
        self.classifier = Dense(3,
                                kernel_initializer=TruncatedNormal(0.02),
                                activation='softmax')

    def call(self, inputs):
        input_ids, attention_mask, token_type_ids=inputs
        outputs = self.bert(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            token_type_ids=token_type_ids)
        cls_token = outputs[1]
        prediction = self.classifier(cls_token)

        return prediction

import tensorflow as tf

model = TFBertForSequenceClassification(model_name)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics='accuracy')

from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
early_stopping = EarlyStopping(
    monitor="val_accuracy", 
    min_delta=0.001,
    patience=2)

model.fit(
    X_train, y_train, epochs=3, batch_size=32, validation_data=(X_valid, y_valid),
    callbacks = [early_stopping]
)

pred = model.predict(X_test)

print(pred.shape)

result = [np.argmax(val) for val in pred]

out = [list(label_idx.keys())[_] for _ in result]
print(out[:3])

now1 = datetime.now()
now_date = now1.strftime("%m%d_%H%M")

sub_sets["label"] = out
sub_sets.to_csv(path + now_date + "jayeouna1.csv", index=True)

'''
loss: 0.3387 - accuracy: 0.8806 - val_loss: 0.4454 - val_accuracy: 0.8351 ㅡ> 0.768
'''
'''
loss: 0.3400 - accuracy: 0.8787 - val_loss: 0.4996 - val_accuracy: 0.8309 ㅡ> 0.779
'''
'''
loss: 0.3737 - accuracy: 0.8664 - val_loss: 0.4840 - val_accuracy: 0.8343 ㅡ> 0.783
'''
'''
loss: 0.2248 - accuracy: 0.9236 - val_loss: 0.5036 - val_accuracy: 0.8347 ㅡ> 0.782
'''
'''
loss: 0.2231 - accuracy: 0.9226 - val_loss: 0.5280 - val_accuracy: 0.8343 ㅡ> 0.8
'''
'''
loss: 0.2172 - accuracy: 0.9254 - val_loss: 0.4657 - val_accuracy: 0.8425
'''