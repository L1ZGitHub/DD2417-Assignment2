import os
import argparse
import time
import numpy as np
from numpy import linalg
import nltk
from collections import defaultdict
from tqdm import tqdm


class RandomIndexing(object):
    def __init__(self, dimension=2000, non_zero=100, non_zero_values=[-1, 1],
                 left_window_size=2, right_window_size=2):

        # Size of the context window
        self.left_window_size = left_window_size
        self.right_window_size = right_window_size

        # Mapping from words to IDs.
        self.word2id = defaultdict(int)
        
        # Mapping from IDs to words.
        self.id2word = []

        # The dimension of the word vectors
        self.dimension = dimension

        # Number of non-zero positions
        self.non_zero = non_zero

        # Possible values at those non-zero positions
        self.non_zero_values = non_zero_values

        # Total number of tokens processed
        self.tokens_processed = 0

        # All datapoints retrieved from the text. A datapoint is a pair (f,c)
        # where f is the ID of the focus word, and c is a list of the IDs of
        # the context words.
        self.datapoints = []

        # Padding at the beginning and end of the token stream
        self.pad_word = '<pad>'

        # Random vectors and context vectors
        self.rv = None
        self.cv = None
        

    #------------------------------------------------------------
    #
    #  Methods for processing all files and mapping all words to ID numbers
    #

    def get_word_id( self, word ) :
        """ 
        Returns the word ID for a given word. If the word has not
        been encountered before, the necessary data structures for
        that word are initialized.
        """
        word = word.lower()
        if word in self.word2id :
            return self.word2id[word]
        
        else : 
            # This word has never been encountered before. Init all necessary
            # data structures.
            latest_new_word = len(self.id2word)
            self.id2word.append(word)
            self.word2id[word] = latest_new_word
 
        return latest_new_word


    def get_context(self, i):
        """
        Returns the context of token no i as a list of word indices.
        
        :param      i:     Index of the focus word in the list of tokens
        :type       i:     int
        """

        # REPLACE THE STATEMENT BELOW WITH YOUR CODE

        if i < self.left_window_size :
            left_context = [self.pad_word] * (self.left_window_size - i) + self.tokens[0:i]
        else :
            left_context = self.tokens[i-self.left_window_size:i]
            
        if i + self.right_window_size >= len(self.tokens) :
            right_context = self.tokens[i+1:] + [self.pad_word] * (self.right_window_size - (len(self.tokens) - i - 1))
        else :
            right_context = self.tokens[i+1:i+1+self.right_window_size]
            
        context = left_context + right_context
        context_ids = [self.get_word_id(word) for word in context]
        
        return context_ids



    def process_files( self, file_or_dir ) :
        """
        This function recursively processes all files in a directory.
        
        Each file is tokenized and the tokens are put in the list
        self.tokens. 
        """
        if os.path.isdir( file_or_dir ) :
            for root,dirs,files in os.walk( file_or_dir ) :
                for file in files :
                    self.process_files( os.path.join(root, file ))
        else :
            print( file_or_dir )
            stream = open( file_or_dir, mode='r', encoding='utf-8', errors='ignore' )
            text = stream.read()
            try :
                self.tokens = nltk.word_tokenize(text) 
            except LookupError :
                nltk.download('punkt')
                self.tokens = nltk.word_tokenize(text)
            for i, token in enumerate(self.tokens) :
                self.tokens_processed += 1
                focus_id = self.get_word_id(token)
                context = self.get_context(i)
                self.datapoints.append( (focus_id, context) )
                if self.tokens_processed % 10000 == 0 :
                    print( 'Processed', "{:,}".format(self.tokens_processed), 'tokens' )

    #
    #  End of methods for processing all files and producing the list of datapoints
    #
    #------------------------------------------------------------

    def create_word_vectors(self):
        """
        This function first creates the random vectors and stores them
        in `self.rv`.
        
        Word embeddings (called `context vectors` in random-indexing
        parlance) are then created by looping through self.datapoints and
        updating the context vectors following the Random Indexing approach,
        i.e., by adding the random vectors of the words appearing in the 
        window around the focus word.
        
        The size of the sliding window is governed by two instance variables
        `self.left_window_size and `self.right_window_size.
        """
        
        # YOUR CODE HERE
        self.rv = np.zeros((len(self.id2word), self.dimension))
        for i in range(len(self.id2word)) :
            non_zero_positions = np.random.choice(self.dimension, self.non_zero, replace=False)
            self.rv[i][non_zero_positions] = np.random.choice(self.non_zero_values, self.non_zero)
        
        self.cv = np.zeros((len(self.id2word), self.dimension))
        for focus_id, context in self.datapoints :
            for context_id in context :
                self.cv[focus_id] += self.rv[context_id]
        

        

    def normalize_word_vectors(self):
        """
        Normalizes all word vectors to unit vectors (of Euclidean length 1)
        """
       
        # YOUR CODE HERE
        for i in range(len(self.cv)) :
            norm = linalg.norm(self.cv[i])
            if norm > 0 :
                self.cv[i] = self.cv[i] / linalg.norm(self.cv[i])
                
        
    def write_word_vectors_to_file( self, filename ) :
        """
        Writes the vectors to file. These are the vectors you would
        export and use in another application.
        """
        with open(filename, 'w', encoding='utf8') as f:
            for idx in range(len(self.id2word)) :
                f.write('{} '.format( self.id2word[idx] ))
                for i in self.cv[idx] :
                    f.write('{} '.format( i ))
                f.write( '\n' )
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Indexing word embeddings')
    parser.add_argument('--file', '-f', type=str,  default='../RandomIndexing/data', help='The files used in the training.')
    parser.add_argument('--left_window_size', '-lws', type=int, default='2', help='Left context window size')
    parser.add_argument('--right_window_size', '-rws', type=int, default='2', help='Right context window size')
    parser.add_argument('--dimension', '-d', type=int, default=2000, help='Dimensionality of word vectors')
    parser.add_argument('--non-zero', '-nz', type=int, default=100, help='Number of non-zero elements')
    parser.add_argument('--output', '-o', type=str, default='vectors.txt', help='The file where the vectors are stored.')
    parser.add_argument('--normalize', '-n', action='store_true', default=False, help='Normalize all vectors to be unit vectors.')

    
    args = parser.parse_args()

    ri = RandomIndexing(
        dimension=args.dimension, left_window_size=args.left_window_size,
        right_window_size=args.right_window_size, non_zero=args.non_zero)
    ri.process_files(args.file)
    print ("Creating word vectors..." )
    ri.create_word_vectors()
    if args.normalize:
        print( "Normalizing..." )
        ri.normalize_word_vectors()
    print( "Saving to file..." )
    ri.write_word_vectors_to_file(args.output)
    print( "Done" )
