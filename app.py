from train import complete_prompt
from flask import Flask, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

app = Flask(__name__)

model = load_model('./model')
tokenizer = Tokenizer()
data = open('poems.txt',encoding="utf8").read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

@app.route('/poem/<username>')
def show_name_poem(name):
	poem = complete_prompt(model, tokenizer, "Dear " + name)
	return render_template('output.html', name=name, poem=poem)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)