from flair.data import Corpus
from flair.models import SequenceTagger
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, StackedEmbeddings, TokenEmbeddings, FlairEmbeddings, ELMoEmbeddings,TransformerWordEmbeddings, BertEmbeddings
from typing import List
import torch

device = None
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
    

def run_splits(embedding_types, embeddings_name):
    embeddings : StackedEmbeddings = StackedEmbeddings(
                                  embeddings=embedding_types)

    for i in range(1,6):
        print('##########')
        print('Split', str(i))
        print('##########')

        # define columns
        columns = {0 : 'text', 1 : 'pos', 2 : 'ner', 3 : 'event', 4 : 'when', 5 : 'who', 6 : 'core', 7 : 'eventtype'}
        # directory where the data resides
        data_folder = '<path_to_splits>/split_' + str(i) + '/'
        # initializing the corpus
        corpus: Corpus = ColumnCorpus(data_folder, columns,
                                      train_file = 'ner_train.csv',
                                      test_file = 'ner_test.csv',
                                      dev_file = 'ner_dev.csv')
        # tag to predict
        tag_type = 'ner'
        # make tag dictionary from the corpus
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
        tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                            embeddings=embeddings,
                                            tag_dictionary=tag_dictionary,
                                            tag_type=tag_type,
                                            use_crf=True)
        print(tagger)

        from flair.trainers import ModelTrainer
        trainer : ModelTrainer = ModelTrainer(tagger, corpus)
        trainer.train(data_folder + '/ner_' + embeddings_name,
                      learning_rate=0.1,
                      mini_batch_size=32,
                      max_epochs=150)

        
embedding_types : List[TokenEmbeddings] = [WordEmbeddings('glove')]
run_splits(embedding_types, 'glove')

embedding_types : List[TokenEmbeddings] = [FlairEmbeddings('news-forward-fast')]
run_splits(embedding_types, 'news-forward-fast')

embedding_types : List[TokenEmbeddings] = [FlairEmbeddings('data/echr_lm_models/news_forward_fast_finetuned_echr/epoch_2.pt')]
run_splits(embedding_types, 'news-forward-fast-finetuned')

word_embeddings = [FlairEmbeddings('data/echr_lm_models/echr_language_model/epoch_4.pt')]
run_splits(word_embeddings, 'flair_echr_13k_lm')

embedding_types : List[TokenEmbeddings] = [TransformerWordEmbeddings('bert-base-cased')]
run_splits(embedding_types, 'bert-base-cased')

embedding_types : List[TokenEmbeddings] = [TransformerWordEmbeddings('distilbert-base-cased')]
run_splits(embedding_types, 'distilbert-base-cased')

word_embeddings = [TransformerWordEmbeddings('data/echr_lm_models/DistilBERT-finetuned')]
run_splits(word_embeddings, 'distilbert-base-cased-finetuned')

word_embeddings = [TransformerWordEmbeddings('data/echr_lm_models/BERT-finetuned')]
run_splits(word_embeddings, 'bert-base-cased-finetuned')

