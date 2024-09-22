import sys
from collections import defaultdict
from collections import Counter
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2023 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1


    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """
    
    
    result = []
    i = 0

    if n == 1:
        start_buffer = []
        result.append(("START",))
        for word in sequence:
            result.append((word,))

        result.append(("STOP",))

    else:   
        start_buffer = []

        while i < (n-1):
            start_buffer.append("START")
            i += 1

        sequence = start_buffer + list(sequence) 
        sequence = tuple(sequence + ["STOP"]) 
        sequence_length = len(sequence)

        k = 0
        j = n

        while j < sequence_length+1:
            collector = []
            while k < j:
                collector.append(sequence[k])
                k += 1
            
            ngram = tuple(collector)
            result.append(ngram)          
            
            k = k - (n-1)
            j += 1
  

    return result


class TrigramModel(object):

    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = Counter() # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        ##Your code here
        self.trainingwordnumber = 0 # Keeps track of total number of words
        self.called = False # Only count words once
        self.sentencenumber = 0

        for sentence in corpus:

            if (self.called == False):
                self.sentencenumber += 1
                
                for word in sentence:              
                    self.trainingwordnumber += 1


            unigram_list = get_ngrams(sentence,1)
            
            for unigram in unigram_list:
                if unigram != (("START",)):
                    self.unigramcounts[unigram] = self.unigramcounts.get(unigram,0) + 1

            
            bigram_list = get_ngrams(sentence,2)

            for bigram in bigram_list:
                self.bigramcounts[bigram] = self.bigramcounts.get(bigram,0) + 1

        
            trigram_list = get_ngrams(sentence,3)

           

            for trigram in trigram_list:
                self.trigramcounts[trigram] = self.trigramcounts.get(trigram,0) + 1

            
        self.called == True

        return
        

    def raw_trigram_probability(self,trigram):
        
        # Returns the raw (unsmoothed) trigram probability
        
        context_count = self.bigramcounts.get((trigram[0],trigram[1]))

        if ((trigram[0],trigram[1]) == (("START","START"))):
            context_count = self.sentencenumber
        
        trigram_count = self.trigramcounts.get(trigram)


        if (trigram_count == None):
            trigram_count = 0

            if (context_count == None):

                return 1 / len(self.lexicon)

        
        return (trigram_count/context_count)

    def raw_bigram_probability(self, bigram):
        
        
        #Returns the raw (unsmoothed) bigram probability
        
        
        bigram_context = self.unigramcounts.get((bigram[0],))

        if ((bigram[0],)) == (("START",)):
            bigram_context = self.sentencenumber
        
        bigram_count = self.bigramcounts.get(bigram)

        if bigram_count == None:
            bigram_count = 0
            

        result = bigram_count / bigram_context

        return result
    
    def raw_unigram_probability(self, unigram):
        
        
        #Returns the raw (unsmoothed) unigram probability.

        if unigram == (("START",)):
            return 0

        result = self.unigramcounts[unigram] / self.trainingwordnumber

        return result

    def generate_sentence(self,t=20): 
 
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        probability = (lambda1 * self.raw_trigram_probability(trigram)) + (lambda2 * self.raw_bigram_probability((trigram[1],trigram[2]))) + (lambda3 * self.raw_unigram_probability((trigram[2],)))
        
        return probability
        
    def sentence_logprob(self, sentence):
        """
        Returns the log probability of an entire sequence.
        """
        result = 0.0

        ngram_list = get_ngrams(sentence,3)

        probability_list = []

        for ngram in ngram_list:
            probability_list.append(math.log2(self.smoothed_trigram_probability(ngram)))


        for probability in probability_list:
            result += probability


        return result 

    def perplexity(self, corpus):
        """
        Returns the log probability of an entire sequence.
        """
        sentence_probabilities = []
        probability_sum = 0.0
        self.testingwordnumber = 0

        for sentence in corpus:
            self.testingwordnumber += 1

            for word in sentence:
                self.testingwordnumber += 1

            sentence_probabilities.append(self.sentence_logprob(sentence))
        


        for probability in sentence_probabilities:
            probability_sum += probability


        return 2**(-(probability_sum/self.testingwordnumber))


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):  #high
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            # .. 

            if pp1 < pp2:
                correct += 1
            total += 1

    
        for f in os.listdir(testdir2): #low
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            # .. 
            if pp2 < pp1:
                correct += 1
            total += 1


        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)


    # Essay scoring experiment: 
    #acc = essay_scoring_experiment("train_high.txt", "train_low.txt", "test_high", "test_low")
    #print(acc)
