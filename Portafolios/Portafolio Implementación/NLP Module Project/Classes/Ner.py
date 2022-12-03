import os

import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings

load_dotenv()
N_EXAMPLES_TO_TRAIN = int(os.environ.get('N_EXAMPLES_TO_TRAIN'))
NER_EPOCHS = int(os.environ.get('NER_EPOCHS'))


class Ner():
    def __init__(self):
        if N_EXAMPLES_TO_TRAIN != 0:
            self.load_custom()

        else:
            self.load_all()

        self.global_config()

    def global_config(self):
        """
        Function that configurates the last things needed for the model to work
        :return:
        """
        self.TAG_TYPE = 'ner'
        self.TAG_DICTIONARY = self.CORPUS.make_label_dictionary(label_type=self.TAG_TYPE, )
        self.EMBEDDING_TYPES = [
            # GloVe embeddings
            WordEmbeddings('glove'),
            # contextual string embeddings, forward
            FlairEmbeddings('news-forward'),
            # contextual string embeddings, backward
            FlairEmbeddings('news-backward'),
        ]
        self.EMBEDDINGS = StackedEmbeddings(embeddings=self.EMBEDDING_TYPES)
        self.TAGGER = SequenceTagger(
            hidden_size=256,
            embeddings=self.EMBEDDINGS,
            tag_dictionary=self.TAG_DICTIONARY,
            tag_type=self.TAG_TYPE,
            use_crf=True,
            allow_unk_predictions=True
        )
        self.LEARNING_RATE = os.environ.get('NER_LEARNING_RATE')
        self.LOSS_PATH = './resources/taggers/example-ner/loss.tsv'

    def load_custom(self):
        """
        Function that loads an specific number of examples to train
        :return:
        """
        # We create the new file with the training data
        with open('./datasets/ner_train.txt', encoding='utf8') as file:
            train_sample = file.readlines()[0:N_EXAMPLES_TO_TRAIN]
            with open('./datasets/ner_train_do.txt', 'w', encoding='utf8') as new_file:
                for line in train_sample:
                    new_file.write(line)
                new_file.close()

        with open('./datasets/ner_test.txt', encoding='utf8') as file:
            test_sample = file.readlines()[0:N_EXAMPLES_TO_TRAIN]
            with open('./datasets/ner_test_do.txt', 'w', encoding='utf8') as new_file:
                for line in test_sample:
                    new_file.write(line)

        with open('./datasets/ner_dev.txt', encoding='utf8') as file:
            dev_sample = file.readlines()[0:N_EXAMPLES_TO_TRAIN]
            with open('./datasets/ner_dev_do.txt', 'w', encoding='utf8') as new_file:
                for line in dev_sample:
                    new_file.write(line)
                new_file.close()

        # Format of the columns
        COLUMNS = {0: 'text', 1: 'ner'}
        # We create the corpus
        self.CORPUS: Corpus = ColumnCorpus(
            data_folder='./datasets',
            column_format=COLUMNS,
            train_file='ner_train_do.txt',
            test_file='ner_test_do.txt',
            dev_file='ner_dev_do.txt'
        )

    def load_all(self):
        """
        Function that loads the whole dataset
        :return:
        """
        # Format of the columns
        COLUMNS = {0: 'text', 1: 'ner'}
        # We create the corpus
        self.CORPUS: Corpus = ColumnCorpus(
            data_folder='./datasets',
            column_format=COLUMNS,
            train_file='ner_train.txt',
            test_file='ner_test.txt',
            dev_file='ner_dev.txt'
        )



    def train(self):
        model_trainer = ModelTrainer(self.TAGGER, self.CORPUS)
        model_trainer.train(
            base_path='./resources/taggers/example-ner',
            learning_rate=0.05,
            mini_batch_size=32,
            max_epochs=NER_EPOCHS
        )

    def plot(self):
        df = pd.read_table(self.LOSS_PATH)
        plt.figure(figsize=(10, 10))
        plt.plot(df['EPOCH'], df['TRAIN_LOSS'], label='Training loss')
        plt.plot(df['EPOCH'], df['DEV_LOSS'], label='Test loss')
        plt.legend(loc='upper right')
        plt.title('Training and test loss')

        plt.show()


if __name__ == '__main__':
    md = Ner()
    md.train()
    md.plot()
