import os
import math
import nltk
import numpy as np
import os.path
import argparse
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors


"""
This file is part of the computer assignments for the course DD2418 Language engineering at KTH.
"""
class VectorTester:

    # Mapping from words to IDs.
    word2id = defaultdict(lambda: None)

    # Mapping from IDs to words.
    id2word = defaultdict(lambda: None)

    # Dimension of word vectors (to be set when the vector file is read).
    dimension = 0

    # Mapping from word IDs to (focus) word vectors. (called vector 
    # to be consistent with the notation in the Glove paper).
    vector = []
 
    # Neighbours
    nbrs = None

    
    def interact(self):
        text = input('> ')
        while text != 'exit':
            text = text.split()
            neighbors = self.find_nearest(text)
            for w, n in zip(text, neighbors):
                print("Neighbors for {}: {}".format(w, n))
            text = input('> ')

            
    def find_nearest(self, words, metric='cosine'):
        """
        Function returning k nearest neighbors with distances for each word
        in `words`
        
        We suggest using nearest neighbors implementation from scikit-learn.
        Carefully check the documentation regarding the parameters passed to
        the algorithm.
    
        Imagine you want to find 5 nearest neighbors for the words "Harry"
        and "Potter" using some distance metric `m`. For that you would need
        to call `self.find_nearest(["Harry", "Potter"], k=5, metric='m')`.

        The output of the function would then be the following list of lists
        of tuples (LLT) (all words and distances are just example values):
    
        [[('Harry', 0.0), ('Hagrid', 0.07), ('Snape', 0.08),
          ('Dumbledore', 0.08), ('Hermione', 0.09)],
         [('Potter', 0.0), ('quickly', 0.21), ('asked', 0.22), ('lied', 0.23),
          ('okay', 0.24)]]
        
        The i-th element of the list would correspond to the k nearest neighbors
        for the i-th word in the `words` input list. Each tuple contains a word
        and a distance. The tuples are sorted either by ascending distance.
        
        :param      words:   Words for the nearest neighbors to be found
        :type       words:   list
        :param      metric:  The similarity/distance metric
        :type       metric:  string
        """
        
        # REPLACE THE STATEMENT BELOW WITH YOUR CODE
        if self.nbrs is None :
            self.nbrs = NearestNeighbors(n_neighbors=5, metric=metric).fit(self.vector)
            
        index = [self.word2id[w] for w in words]
        distances, indices = self.nbrs.kneighbors([self.vector[i] for i in index])
        neighbors = []
        for i in range(len(words)) :
            neighbors.append([(self.id2word[indices[i][j]], distances[i][j]) for j in range(len(indices[i]))])
            
        return neighbors


    # Reads the vectors from file
    def read_vectors(self, fname):
        i = 0
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                data = line.split()
                w = data[0]
                vec = np.array([float(x) for x in data[1:]])
                self.id2word[i] = w
                self.word2id[w] = i
                self.vector.append(vec)
                i += 1
        f.close()
        self.dimension = len( self.vector[0] )

       
def main() :
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Vector tester')
    parser.add_argument('--file', '-f', type=str,  required=True, help='The files used in the training.')

    arguments = parser.parse_args()  
    
    vt = VectorTester()
    vt.read_vectors( arguments.file )
    vt.interact()


        
if __name__ == '__main__' :
    main()    

