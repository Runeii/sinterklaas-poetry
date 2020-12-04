from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model#, Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku 
import numpy as np

def tokenize():
	tokenizer = Tokenizer()
	data = open('poems.txt',encoding="utf8").read()

	corpus = data.lower().split("\n")

	tokenizer.fit_on_texts(corpus)

	# Train data
	input_sequences = []
	for line in corpus:
		token_list = tokenizer.texts_to_sequences([line])[0]
		for i in range(1, len(token_list)):
			n_gram_sequence = token_list[:i+1]
			input_sequences.append(n_gram_sequence)


	max_sequence_len = max([len(x) for x in input_sequences])
	input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

	return tokenizer, input_sequences, max_sequence_len

def complete_prompt(model, tokenize_result, prompt):
	tokenizer, input_sequences, max_sequence_len = tokenize_result
	seed_text = prompt
	next_words = 50
	
	for _ in range(next_words):
		token_list = tokenizer.texts_to_sequences([seed_text])[0]
		token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
		predicted = model.predict_classes(token_list, verbose=0)
		output_word = ""
		for word, index in tokenizer.word_index.items():
			if index == predicted:
				output_word = word
				break
		seed_text += " " + output_word
	return seed_text

def train():
	# Tokenize
	tokenize_result = tokenize()
	tokenizer, input_sequences, max_sequence_len = tokenize_result
	total_words = len(tokenizer.word_index) + 1
	predictors, label = input_sequences[:,:-1],input_sequences[:,-1]

	label = ku.to_categorical(label, num_classes=total_words)

	# Model building

	# print("Creating new model")
	# model = Sequential()
	# model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
	# model.add(Bidirectional(LSTM(150, return_sequences = True)))
	# model.add(Dropout(0.2))
	# model.add(LSTM(100))
	# model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
	# model.add(Dense(total_words, activation='softmax'))
	# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


	i = 0
	while True:
		i += 1
		print('=============Starting run"================')
		print(i)
		model = load_model('./model')
		print("Loaded saved model")
		#if x == 0:
		#	print(model.summary())
		result = complete_prompt(model, tokenize_result, "Dear Leigh")
		print(result)
		model.fit(predictors, label, epochs=100, verbose=1, use_multiprocessing=True)
		model.save('model')

if __name__ == '__main__':
	train()