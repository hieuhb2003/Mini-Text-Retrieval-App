from flask import Flask, render_template, request, redirect, url_for
from datasets import load_dataset
from model.BoW import BoWProcessor
from model.GLOVE import GloVeProcessor
from model.BERT import BertProccessor

app = Flask(__name__)

# Load dataset and initialize processors
dataset = load_dataset('ms_marco', 'v1.1')['test']  # Load a subset for demo purposes
bow_processor = BoWProcessor(dataset)
glove_processor = GloVeProcessor(dataset, 'GloVe_300.json')  # Update with correct path
bert_processor = BertProccessor(dataset)

# Store the current rankings
current_rankings = []
current_corpus = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        processor_type = request.form['processor']
        topk = 10

        if processor_type == "BoW":
            processor = bow_processor
        elif processor_type == "GloVe":
            processor = glove_processor
        else:
            processor = bert_processor

        global current_rankings, current_corpus
        current_rankings = processor.ranking(query, topk)
        current_corpus = processor.corpus

        return redirect(url_for('results'))

    return render_template('index.html')

@app.route('/results')
def results():
    global current_rankings, current_corpus
    return render_template('results.html', rankings=current_rankings, corpus=current_corpus, enumerate=enumerate)

@app.route('/document/<int:doc_idx>')
def document(doc_idx):
    global current_rankings, current_corpus
    score = current_rankings[doc_idx][1]
    content = current_corpus[current_rankings[doc_idx][0]]
    return render_template('document.html', doc_idx=doc_idx, score=score, content=content)

if __name__ == "__main__":
    app.run(debug=True)
