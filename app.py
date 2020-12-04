from train import complete_prompt, tokenize
from flask import Flask, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('./model')

tokenize_result = tokenize()

@app.route('/<name>')
def show_name_poem(name):
	poem = complete_prompt(model, tokenize_result, "Dear " + name)
	return render_template('output.html', name=name, poem=poem)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)