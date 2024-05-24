import tkinter as tk
from datasets import load_dataset
from BoW import BoWProcessor
from GLOVE import GloVeProcessor  # Import GloVeProcessor
from Bert_Model import BertProccessor

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Text Retrieval App")

        self.left_frame = tk.Frame(self)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.right_frame = tk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.option_var = tk.StringVar(value="BoW")

        self.option_menu = tk.OptionMenu(self.left_frame, self.option_var, "BoW", "GloVe", "Bert")
        self.option_menu.pack(fill=tk.X, padx=5, pady=5)

        self.search_bar = SearchBar(self.left_frame, self.search)
        self.search_bar.pack(fill=tk.X, padx=5, pady=5)

        self.result_box = ResultBox(self.left_frame, self.show_document)
        self.result_box.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.document_display = DocumentDisplay(self.right_frame)
        self.document_display.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Load dataset and initialize processors
        dataset = load_dataset('ms_marco', 'v1.1')['test']  # Load a subset for demo purposes
        self.bow_processor = BoWProcessor(dataset)
        self.glove_processor = GloVeProcessor(dataset, 'GloVe_300.json')  # Update with correct path
        self.bert_processor = BertProccessor(dataset)
        # Store the current rankings
        self.current_rankings = []

    def search(self, query):
        topk = 10
        if self.option_var.get() == "BoW":
            processor = self.bow_processor
        elif self.option_var.get() == "GloVe":
            processor = self.glove_processor
        else:
            processor = self.bert_processor
        #processor = self.bow_processor if self.option_var.get() == "BoW" else self.glove_processor
        self.current_rankings = processor.ranking(query, topk)
        self.result_box.update(self.current_rankings, processor.corpus)
    #
    # def show_document(self, index):
    #     processor = self.bow_processor if self.option_var.get() == "BoW" else self.glove_processor
    #     doc_idx, score = self.current_rankings[index]
    #     content = processor.corpus[doc_idx]
    #     self.document_display.update(f"Document {doc_idx}", content, score)

    def show_document(self, index):
        if self.option_var.get() == "BoW":
            processor = self.bow_processor
        elif self.option_var.get() == "GloVe":
            processor = self.glove_processor
        else:
            processor = self.bert_processor

        doc_idx, score = self.current_rankings[index]
        content = processor.corpus[doc_idx]
        self.document_display.update(f"Document {doc_idx}", content, score)


class SearchBar(tk.Frame):
    def __init__(self, parent, search_callback):
        super().__init__(parent)
        self.search_callback = search_callback

        self.entry = tk.Entry(self)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)

        self.button = tk.Button(self, text="Search", command=self.search)
        self.button.pack(side=tk.LEFT, padx=5, pady=5)

    def search(self):
        query = self.entry.get().strip()
        if query:
            self.search_callback(query)

class ResultBox(tk.Listbox):
    def __init__(self, parent, select_callback):
        super().__init__(parent)
        self.select_callback = select_callback
        self.bind('<<ListboxSelect>>', self.on_select)

    def on_select(self, event):
        index = self.curselection()
        if index:
            self.select_callback(index[0])

    def update(self, rankings, corpus):
        self.delete(0, tk.END)
        for idx, (doc_idx, score) in enumerate(rankings):
            snippet = corpus[doc_idx][:50]  # Display the first 50 characters as a snippet
            self.insert(tk.END, f"Document {doc_idx}: Score {score:.4f} - {snippet}...")

class DocumentDisplay(tk.Text):
    def __init__(self, parent):
        super().__init__(parent)
        self.config(state=tk.DISABLED)

    def update(self, title, content, score):
        self.config(state=tk.NORMAL)
        self.delete('1.0', tk.END)
        self.insert(tk.END, f"Title: {title}\n\n")
        self.insert(tk.END, f"Score: {score:.4f}\n\n")
        self.insert(tk.END, f"Content:\n{content}")
        self.config(state=tk.DISABLED)

if __name__ == "__main__":
    app = App()
    app.mainloop()
