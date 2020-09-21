from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentLSTMEmbeddings, TransformerWordEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.datasets import CSVClassificationCorpus, ClassificationCorpus
from pathlib import Path
from flair.data import Dictionary
import torch

device = None
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def run_splits(word_embeddings, embeddings_name):
    for i in range(1,6):
        print('##########')
        print('Split', str(i))
        print('##########')


        data_folder = '<path_to_splits>/split_' + str(i) + '/'
        corpus = ClassificationCorpus(data_folder,
                                              test_file='test.csv',
                                              dev_file='dev.csv',
                                              train_file='train.csv')

        document_embeddings = DocumentLSTMEmbeddings(
          word_embeddings, 
          hidden_size=512, 
          reproject_words=True, 
          reproject_words_dimension=256
          )

        classifier = TextClassifier(
            document_embeddings, 
            label_dictionary=corpus.make_label_dictionary(), 
            multi_label = False
            )

        trainer = ModelTrainer(classifier, corpus)
        trainer.train(data_folder + '/' + embeddings_name, max_epochs=150)


word_embeddings = [WordEmbeddings('glove')]
run_splits(word_embeddings, 'glove')

word_embeddings = [FlairEmbeddings('news-forward-fast')]
run_splits(word_embeddings, 'news-forward-fast')

word_embeddings = [FlairEmbeddings('data/echr_lm_models/news_forward_fast_finetuned_echr/epoch_2.pt')]
run_splits(word_embeddings, 'news-forward-fast-finetuned')

word_embeddings = [FlairEmbeddings('data/echr_lm_models/echr_language_model/epoch_4.pt')]
run_splits(word_embeddings, 'flair_echr_13k_lm')

word_embeddings = [TransformerWordEmbeddings('bert-base-cased')]
run_splits(word_embeddings, 'bert-base-cased')

word_embeddings = [TransformerWordEmbeddings('distilbert-base-cased')]
run_splits(word_embeddings, 'distilbert-base-cased')

word_embeddings = [TransformerWordEmbeddings('data/echr_lm_models/DistilBERT-finetuned')]
run_splits(word_embeddings, 'distilbert-base-cased-finetuned')

word_embeddings = [TransformerWordEmbeddings('data/echr_lm_models/BERT-finetuned')]
run_splits(word_embeddings, 'bert-base-cased-finetuned')



