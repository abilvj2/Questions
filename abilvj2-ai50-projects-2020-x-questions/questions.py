import nltk
import sys
import os
import string
import math
from collections import Counter
FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    path = os.getcwd()
    path = os.path.join(path, directory)
 
    dir_list = os.listdir(path)

    return_dict = {} 

    for filename in dir_list:
        filename_path = os.path.join(path, filename)
        with open(filename_path, encoding="utf-8") as f:
            contents = f.read()
            return_dict[filename] = contents
    return return_dict
      
def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    wordlist = nltk.word_tokenize(document)

    size = len(wordlist) 
    ls_retn = []
    for i in range(size):
        wordlist[i] = wordlist[i].lower() 
        if wordlist[i] not in string.punctuation and wordlist[i] not in nltk.corpus.stopwords.words("english"): 
            ls_retn.append(wordlist[i])

    return ls_retn


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    big_set = set() 
    file_num = len(documents)

    for file_key in documents:

        for word in documents[file_key]: 
            big_set.add(word)

    dict_return = {}

    for word in big_set:
        count = 0
        for file_key_1 in documents:
            if word in documents[file_key_1]:
                count += 1
        dict_return[word] = math.log(file_num / count)
    return dict_return




def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    def tf_idf(filename):
        cnt = Counter(files[filename]) 
        sum = 0
        for word in query: 
            if word in cnt:
                sum += idfs[word] * cnt[word]
        return sum
    ls = []
    for key in files: 
        ls.append(key)
    ls.sort(key=tf_idf, reverse=True)

    return ls[: n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    def sum_idf(sentence):
        sum = 0
        for word in query:
            if word in sentences[sentence]:
                sum += idfs[word]
        return sum

    def query_density(sentence): 
        count = 0
        for word in sentences[sentence]:
            if word in query:
                count += 1
                return count / len(sentences[sentence])

    ls = []
    for sentence in sentences:
        ls.append(sentence)
    ls.sort (key=lambda x: (sum_idf(x), query_density(x)), reverse=True)
    return ls[: n]


if __name__ == "__main__":
    main()
