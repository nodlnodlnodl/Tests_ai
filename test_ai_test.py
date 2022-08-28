import numpy as np
import re

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


from test_ai import multiply_string

model = keras.models.load_model('vk_neyromodel')


#def test_train():
#    test_train.readline()
#    while True:
#        click_streem = test_train.readline()
#        if not click_stream:
#            break
#        texts, res = multiply_string(click_streem, answer=True)
#        data = tokenizer.texts_to_sequences([texts])
#        data_pad = pad_sequences(data, maxlen=max_text_len)
#        print(sequence_to_text(data[0]))
#        res = model.predict(data_pad)
#        print(res, np.argmax(res), sep='\n')

#model.open('vk_neyromodel')
def oneorzero():
    output_f = open('answer.csv', 'w')
    f = open('./dataset/test.csv', 'r')
    f.readline()
    output_f.write('CLIENT_ID,RETRO_DT,DEF\n')
    for _ in range(0,10000):
        click_streem = f.readline()
        if not click_streem:
            break
        cid, dt, text = multiply_string(click_streem, answer= True, csid=True, dtadd=True)
        data = tokenizer.texts_to_sequences([text])
        dtpad = pad_sequences(data)
        res = model.predict(dtpad)  
        print(f'res: {res}\tanswer: {answer}')
        output_f.write(f'{cid},{dt},{res}\n')
oneorzero()
