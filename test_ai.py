import numpy as np
import re

from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences


hash_pattern = r'[a-z0-9]{32} +\d'
word_pattern = r'\w{2,} +\d'

def multiply_string(click_streem, answer = False, csid=False, dtadd=False):
    csid, dt, string_of_words, string_of_hash = click_streem.split(',')
    words = ''
    temp_words = re.findall(word_pattern, string_of_words)
    for word in temp_words:
        letters, num = word.split()
        for _ in (0, int(num)):
            words += f'{letters}, '
    hashes = ''
    temp_hashes = re.findall(hash_pattern, string_of_hash)
    for h in temp_hashes:
        letters, num = h.split()
        for _ in (0, int(num)):
            hashes += f'{letters}, '
    hashes += words
    if answer:
        return csid, dt, hashes
          



f = open('dataset/train_smal_pos.csv', 'r', encoding='utf-8')
f.readline()                                        # skip meta information
while True:
    raw_text = f.readline()
    if not raw_text:
        break
    texts_true = multiply_string(raw_text)
    

f = open('dataset/train_smal_neg.csv', 'r', encoding='utf-8')
f.readline()                                        # skip meta information
while True:
    raw_text = f.readline()
    if not raw_text:
        break
    texts_false = multiply_string(raw_text)

texts = texts_true + texts_false
count_true = len(texts_true)
count_false = len(texts_false)
total_lines = count_true + count_false
print(count_true, count_false, total_lines)


maxWordsCount = 1000
tokenizer = Tokenizer(num_words=maxWordsCount, lower=True, 
        filters=' ', split=', ', char_level=False)
tokenizer.fit_on_texts([texts])

dist = list(tokenizer.word_counts.items())
print(dist[:10])
print(texts[0][:100])


max_text_len = 1000
data = tokenizer.texts_to_sequences(texts)
data_pad = pad_sequences(data, maxlen=max_text_len)
print(data_pad)

print( list(tokenizer.word_index.items()) )


X = data_pad
Y = np.array([[1, 0]]*count_true + [[0, 1]]*count_false)
print(X.shape, Y.shape)

indeces = np.random.choice(X.shape[0], size=X.shape[0], replace=False)
X = X[indeces]
Y = Y[indeces]


model = Sequential()
model.add(Embedding(maxWordsCount, 128, input_length = max_text_len))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(0.0001))

history = model.fit(X, Y, batch_size=2048, epochs=5)

reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))


model.save_weights('vk_weights')
model.save('vk_neyromodel')


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

