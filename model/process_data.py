'''
    Process stop words and labels
'''


from nltk.corpus  import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pandas as pd
import pickle as pk
import numpy as np
import os
# testing
import pdb


class ProcessData:

    def __init__(self, data):
        self.data = data


    def createTrainData(self):
        '''
            process data with the following steps:
                (1) remove stop words and common punctuation, set to lower case
                (2) tokenize words into integers, store index in tokenizer
                (3) create one hot vector for data label
        '''
        # uncomment to include stop num_words
        # self.removeStopWords()
        padded_sequence, tokenizer = self.createTokenizeData()
        labels = self.createDataLabel()
        return padded_sequence, tokenizer, labels

    def removeStopWords(self):
        '''
            remove the stop words and common punctuation from each tweet
            and set string to lower-case
            i.e. "My flight really was the worse." transformed to
            "flight really worse."
        '''
        data = self.data['text']

        stop=stopwords.words('english')
        punctuation = [',','.',';']
        print('Removing stop words and punctuations...')
        for p in punctuation:
            data = data.map(lambda x: str(x).replace(p, ''))
        text=[]
        data = data.map(lambda x: ' '.join(word for word in 
                                    str(x).lower().split() 
                                       if word not in set(stop) ))
        self.data['text'] = data

    def createTokenizeData(self):
        '''
            Tokenize sequence into integers
        '''
        texts =  [str(string) for string in self.data.text.values]
        # set high threshold for max number of words to keep
        tokenizer = Tokenizer(num_words=10000)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        word_index = tokenizer.word_index
        padded_sequence = pad_sequences(sequences)
        return padded_sequence, tokenizer

    def createDataLabel(self):
        '''
            create data labels labels
        '''
        label_index = {'neutral':1,'negative':2,'positive':0}
        y_label = self.data['airline_sentiment'].map(label_index)
        labels = to_categorical(np.asarray(y_label))
        return labels


class createFTEmbedding():

    def __init__(self):
        '''
            Initiate the path to fasttext vector files stored locally
        '''
        pickle_filename = '~/FastData/wiki.en/wiki.en.pkl'
        self.pickle_path = os.path.expanduser(pickle_filename)


    def processDict(self, tokenizer):
        '''
            Load the fast text word dict and then
            translate to an emedding matrix
        '''
        word_vec =  self.loadWordDict()
        tokenized_word_index = tokenizer.word_index
        self.creatEmbedMatrix(tokenized_word_index, word_vec)


    def loadWordDict(self):
        '''
            Load the word dictionary
        '''
        pickle_reader = open(self.pickle_path, 'rb')
        word_vec = pk.load(pickle_reader)
        return word_vec


    def creatEmbedMatrix(self, word_index, ft_word_vec ):
        '''
            convert the tokenized word indices
            into an embedding matrix where the row corresponds to the
            tokenized index, and the vector directly from fasttext
        '''
        # create an embedding matrix
        num_words = len(word_index) + 1
        embedding_dim = len(ft_word_vec.get('.'))
        embedding_matrix = np.zeros((num_words, embedding_dim))

        for word, i in word_index.items():
            embedding_vector = ft_word_vec.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        self.embedding_matrix = embedding_matrix



def testStopWords():
    '''
        Unit test for stop words
    '''
    df = pd.DataFrame({'text': ['Perhaps, we are all pig .',
            'I have nowhere! ;']})
    print(df)
    process = ProcessData(df)
    process.removeStopWords()
    print(process.data.text)
    assert process.data.text[0] == 'perhaps pig'
    assert process.data.text[1] == 'nowhere!'
    print('PASS PUNCTUATION AND STOP WORD REMOVAL!')


def testTokenizeWords():
    '''
        Unit test fasttext embedding class and functions
    '''
    df = pd.DataFrame({'text': ['Perhaps, we are all pig .',
            'I have nowhere! ;']})
    process = ProcessData(df)
    padded_sequence, tokenizer = process.createTokenizeData()
    ft = createFTEmbedding()
    ft.processDict(tokenizer)
    # check that for num_words = number of distinct words in data
    # and embed_dim = embedding size of the pretrained embeddings
    # matrix has dimension (num_words+1, embed_dim)
    assert ft.embedding_matrix.shape[0] == 9
    assert ft.embedding_matrix.shape[1] == 300
    print('PASS EMBEDDING MATRIX!')


# Class create embedding matrix from fasttext

if __name__ == "__main__":
    testStopWords()
    testTokenizeWords()



