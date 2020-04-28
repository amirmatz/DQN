import fasttext

model = fasttext.train_unsupervised("text.txt",thread=1, model='skipgram')
model.save_model("fast_text.bin")