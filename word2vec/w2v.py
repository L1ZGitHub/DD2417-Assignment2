import os
import time
import nltk
import json
import argparse
import numpy as np
import numpy.random as rand
from collections import defaultdict
from tqdm import tqdm



class Word2Vec(object):
    def __init__(self, dimension=50, left_window_size=2, right_window_size=2, nsample=10,
                 learning_rate=0.025, epochs=5, use_corrected=True, use_lr_scheduling=True,
                 output="vectors.txt" ):

        # Size of the context window
        self.left_window_size = left_window_size
        self.right_window_size = right_window_size

        # Mapping from words to IDs.
        self.word2id = defaultdict(int)
        
        # Mapping from IDs to words.
        self.id2word = []

        # Mapping from word IDs to (focus) word vectors. (called w_vector 
        # to be consistent with the notation in the lecture).
        self.w_vector = []

        # Mapping from word IDs to (context) word vectors (called w_tilde_vector
        # to be consistent with the notation in the lecture)
        self.w_tilde_vector = []

        # Total number of tokens processed
        self.tokens_processed = 0

        # Number of occurrences of each unique word
        self.freq = defaultdict(int)

        # Dimension of word vectors.
        self.dimension = dimension

        # Initial learning rate.
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate

        # All datapoints retrieved from the text. A datapoint is a pair (f,c)
        # where f is the ID of the focus word, and c is a list of the IDs of
        # the context words.
        self.datapoints = []

        # Number of negative samples in each iteration
        self.no_of_neg_samples = nsample

        # Number of epochs through all datapoints
        self.epochs = epochs
        self.current_epoch = 0

        # Use (or don't use) the corrected sampling distribution for negative examples
        self.use_corrected = use_corrected

        # Use (or don't use) learning rate scheduling 
        self.use_lr_scheduling = use_lr_scheduling

        # Padding at the beginning and end of the token stream
        self.pad_word = '<pad>'

        # Name of the file to store the word vectors
        self.outputfile = output

        # Temporary file used for storing the model
        self.temp_file = "temp__.txt"


    #------------------------------------------------------------
    #
    #  Methods for processing all files and producing the list of datapoints
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
            
            # Initialize arrays with random numbers in [-0.5,0.5].
            w = rand.rand(self.dimension)-0.5
            self.w_vector.append(w)
            w_tilde = rand.rand(self.dimension)-0.5
            self.w_tilde_vector.append(w_tilde)
            return latest_new_word

        
    def get_context(self, i):
        """
        Returns the context of token no i as a list of word indices.
        
        :param      i:     Index of the focus word in the list of tokens
        :type       i:     int
        """

        # USE YOUR CODE FROM THE RANDOM INDEXING TASK

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
                self.freq[focus_id] += 1
                if self.tokens_processed % 10000 == 0 :
                    print( 'Processed', "{:,}".format(self.tokens_processed), 'tokens' )

    #
    #  End of methods for processing all files and producing the list of datapoints
    #
    #------------------------------------------------------------
                   

    #------------------------------------------------------------
    #
    #   Training
    #

    def compute_sampling_distributions(self) :

        # REPLACE THE STATEMENT BELOW WITH YOUR CODE

        P_u, P_w = [], []
        for i in range(len(self.id2word)) :
            P_u.append(self.freq[i] / self.tokens_processed)
        
        P_w = np.array(P_u)**0.75 / np.sum(np.array(P_u)**0.75)
        
        # P_u is the unigram distribution,
        # P_w is the modified unigram distribution
        return P_u, P_w
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    def negative_sampling(self, number, focus, pos, distribution):
        """
        Samples `number` of negatives examples. The words `focus` and `pos`
        are taboo, i.e., those should not be selected.
        
        :param      number:     The number of negative examples to be sampled
        :param      focus:      The ID of the current focus word
        :param      pos:        The ID of the current positive example
        :param      distribution A list of sampling probabilities (one for each unique word)
        """

        # REPLACE THE STATEMENT BELOW WITH YOUR CODE
        samples = []
        while len(samples) < number:
            candidates = np.random.choice(len(self.id2word), size=number, p=distribution)
            for c in candidates:
                if c != focus and c != pos:
                    samples.append(c)
                    if len(samples) == number:
                        break
        return samples


    def train(self):
        """
        Performs the training of the word2vec skip-gram model
        """
        print( f"Dataset contains {len(self.datapoints)} datapoints" )

        P_u, P_w = self.compute_sampling_distributions()
        distribution = P_w if self.use_corrected else P_u

        while self.current_epoch < self.epochs:
            for i in tqdm(range(len(self.datapoints))):


                # YOUR CODE HERE
                if self.use_lr_scheduling:
                    self.learning_rate = self.initial_learning_rate * max(0.0001, 1 - ( (i + self.current_epoch * len(self.datapoints) ) / (self.epochs * ( len(self.datapoints) + 1 ) ) ) )
                    
                context = self.datapoints[i][1]
                for pos in context :

                    focus_vector = self.w_vector[self.datapoints[i][0]]
                    pos_vector = self.w_tilde_vector[pos]
                    score = np.dot(focus_vector, pos_vector)
                    sigmoid_score = self.sigmoid(score)
                    grad_focus = (sigmoid_score - 1) * pos_vector
                    grad_pos = (sigmoid_score - 1) * focus_vector
                    self.w_vector[self.datapoints[i][0]] -= self.learning_rate * grad_focus
                    self.w_tilde_vector[pos] -= self.learning_rate * grad_pos

                    negative_samples = self.negative_sampling(self.no_of_neg_samples, self.datapoints[i][0], pos, distribution)
                    
                    for neg in negative_samples :
                        neg_vector = self.w_tilde_vector[neg]
                        score = np.dot(focus_vector, neg_vector)
                        sigmoid_score = self.sigmoid(score)
                        grad_focus = sigmoid_score * neg_vector
                        grad_neg = sigmoid_score * focus_vector
                        self.w_vector[self.datapoints[i][0]] -= self.learning_rate * grad_focus
                        self.w_tilde_vector[neg] -= self.learning_rate * grad_neg

                
            if self.use_lr_scheduling:
                print("[lr={}] Epoch {} is finished!".format(self.learning_rate, self.current_epoch + 1))
            else:
                print("Epoch {} is finished!".format(self.current_epoch + 1))
            # Write to file after each epoch.
            self.current_epoch += 1
            self.write_word_vectors_to_file( self.outputfile )
            self.write_temp_file( self.temp_file )
            
    #
    #  End of training
    #
    #-------------------------------------------------------

    #-------------------------------------------------------
    #
    #  I/O
    #

    def write_word_vectors_to_file( self, filename ) :
        """
        Writes the vectors to file. These are the vectors you would
        export and use in another application.
        """
        with open(filename, 'w') as f:
            for idx in range(len(self.id2word)) :
                f.write('{} '.format( self.id2word[idx] ))
                for i in self.w_vector[idx] :
                    f.write('{} '.format( i ))
                f.write( '\n' )
        f.close()

        
    def write_temp_file( self, filename ) :
        """
        Saves the state of the computation to file, so that
        training can be resumed later.
        """
        with open(filename, 'w') as f:
            data = {"initial_learning_rate":self.initial_learning_rate,
                    "learning_rate":self.learning_rate,
                    "dimension":self.dimension,
                    "no_of_neg_samples":self.no_of_neg_samples,
                    "epochs":self.epochs,
                    "current_epoch":self.current_epoch,
                    "use_corrected":self.use_corrected,
                    "use_lr_scheduling":self.use_lr_scheduling,
                    "left_window_size":self.left_window_size, 
                    "right_window_size":self.right_window_size} 
            json.dump( data, f ) 
            f.write( '\n' )
            for idx in range(len(self.id2word)) :
                f.write('{} '.format( self.id2word[idx] ))
                for i in list(self.w_vector[idx]) :
                    f.write('{} '.format( i ))
                for i in list(self.w_tilde_vector[idx]) :
                    f.write('{} '.format( i ))
                f.write( '\n' )
        f.close()


    def read_temp_file(self, fname):
        """
        Reads the partially trained model from file, so
        that training can be resumed.
        """
        try: 
            with open(fname) as f:
                data = json.loads(f.readline().strip())
                self.initial_learning_rate = float(data.get("initial_learning_rate", 0.05))
                self.learning_rate = float(data.get("learning_rate", 0.05))
                self.dimension = int(data.get("dimension",50))
                self.no_of_neg_samples = int(data.get("no_of_neg_sample", 10))
                self.epochs = int(data.get("epochs", 5))
                self.current_epoch = int(data.get("current_epoch", 0))
                self.use_corrected = bool(data.get("use_corrected", True))
                self.use_lr_scheduling = bool(data.get("use_lr_scheduling", True))
                self.left_window_size = int(data.get("left_window_size", 2))
                self.right_window_size = int(data.get("right_window_size", 2))
                for line in f:
                    data = line.split()
                    w = data[0]
                    vec = np.array([float(x) for x in data[1:self.dimension+1]])
                    self.word2id[w] = len(self.id2word)
                    self.id2word.append(w)
                    self.w_vector.append(vec)
                    vec = np.array([float(x) for x in data[self.dimension+1:]])
                    self.w_tilde_vector.append(vec)
            f.close()
            print( "Read temp file, resuming training." )
        except FileNotFoundError:
            print( "Could not find temp file, starting training from scratch." )
        except ValueError:
            print( "Error reading temp file, starting training from scratch." )
       
    #
    #  End of I/O
    #
    #-------------------------------------------------------



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='word2vec embeddings toolkit')
    parser.add_argument('--file', '-f', type=str,  default='../RandomIndexing/data', help='The files used in the training.')
    parser.add_argument('--left_window_size', '-lws', type=int, default='2', help='Left context window size')
    parser.add_argument('--right_window_size', '-rws', type=int, default='2', help='Right context window size')
    parser.add_argument('--continue_training', '-c', action='store_true', default=False, help='Continues training from where it was left off.')
    parser.add_argument('--output', '-o', type=str, default='vectors.txt', help='The file where the vectors are stored.')
    parser.add_argument('--dimension', '-d', type=int, default=50, help='Dimensionality of word vectors')
    parser.add_argument('--negative_samples', '-neg', type=int, default=10, help='Number of negative samples')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.025, help='Initial learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Number of epochs')
    parser.add_argument('--use-corrected', '-uc', action='store_true', default=True,
                        help="Use the corrected unigram distribution for negative sampling")
    parser.add_argument('--use-learning-rate-scheduling', '-ulrs', action='store_true', default=True,
                        help="Use learning rate scheduling")
    args = parser.parse_args()

    w2v = Word2Vec(
        dimension=args.dimension, left_window_size=args.left_window_size,
        right_window_size=args.right_window_size, output=args.output,
        nsample=args.negative_samples, learning_rate=args.learning_rate, epochs=args.epochs,
        use_corrected=args.use_corrected, use_lr_scheduling=args.use_learning_rate_scheduling
    )
    if args.continue_training:
        w2v.read_temp_file( w2v.temp_file )
    w2v.process_files( args.file )
    print( 'Processed', "{:,}".format(w2v.tokens_processed), 'tokens' )
    print( 'Found', len(w2v.word2id), 'unique words' )
    w2v.train()




