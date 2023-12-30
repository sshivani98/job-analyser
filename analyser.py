'''
This tool uses NLP with NLTK and Spacy to identify the required skills from a given job description

Also does the same when multiple files are provided
'''
# imports
import argparse
import numpy as np
from glob import glob
import typing
import json
# import pandas as pd

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
nltk_stopwords = stopwords.words('english') #english lang 

from gensim.models import KeyedVectors

# Utility functions
def read_file(file_path: str) -> str:
    '''
    Reads a file given a file path
    
    Args:
        file_path - path to the file
    Returns:
        content of the file
    '''
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def write_file(file_path: str, content: str) -> None:
    '''
    Writes a content to a file given a path

    Args:
        file_path - path to the file to be written
        content - content to be written
    Returns:
        None
    '''
    with open(file_path, 'w') as file:
        file.write(content)

def sort_dict_on_values(d: dict) -> dict:
    '''
    Sorts a dictionary based on values

    Args:
        d - input dictionary to be sorted
    Returns:
        sorted dict
    '''
    return {k: d[k] for k in sorted(d, key=lambda k: d[k], reverse=True)}

def extract_unique_words(w2v_model: typing.Any, words_list: typing.List[str]) -> typing.List[str]:
    '''
    A function that accepts the word2vec model and list of words. Inside it,
    Compares each word in the list against each other and removes one of the duplicates
        if the similarity is >=0.75
    It returns a list with duplicates removed

    Args:
        w2v_model - word2vec model from gensim
        words_list - list of words
    Returns:
        a list of strings containing the most unique words
    '''
    words_list_copy = words_list
    removed = list()
    for idx1, word1 in enumerate(words_list):
        for idx2, word2 in enumerate(words_list[idx1+1:]):
            if w2v_model.similarity(word1, word2) >= 0.7:
                words_list_copy.remove(word2)
                removed.append(word2)
    return words_list_copy


def extract_word_counts(
    string: str,
    w2v_model: typing.Any,
    only_keep_repeated: bool = False,
    tech_terms_file: str = 'probably_tech_terms.txt'
) -> dict:
    '''
    Cleans, removes stop-words and retains the most unique words in terms of meaning, \
        returns a dict containing words and their respective counts, sorted based on counts (values)

    Args:
        string - input job description as a string
        only_keep_repeated - only retain the most repeated skills/ terms/ tools
        tech_terms_file - path to a file containing a list of tech terms
    Returns:
        A dict containing word-count 
    '''

    #1. To remove unwanted special characters 
    special_chars = ",./\n&!@#$%^*()><:;'?\"\\~`-_+={[]}|"
    for i in special_chars:
        string = string.replace(i, " ")
    string = string.lower()
    strings_list = string.split()
    
    #2. To remove stopwords using NLTK
    for word in nltk_stopwords:
        stopwords_count = strings_list.count(word)
        j = 0
        while j < stopwords_count:
            strings_list.remove(word)
            j+=1

    #3. Create a Dict to map word and its count 
    words, counts = np.unique(strings_list, return_counts=True)
    word_counts = {}
    for i in range(len(words)):
        word_counts[words[i]] = counts[i]
    
    #4. Sort the Dict to find max occuring word in order
    word_counts_sorted_dict = {}
    w_words = list(word_counts.keys())
    w_counts = list(word_counts.values())
    w_counts_sorted = sorted(w_counts, reverse=True)

    for c in w_counts_sorted:
        pos = w_counts.index(c)
        extracted_word = w_words[pos]
        w_counts.remove(c)
        w_words.remove(extracted_word)

        if only_keep_repeated and c < 2:
            pass
        else:
            word_counts_sorted_dict[extracted_word] = c
    
    all_selected_words = list(word_counts_sorted_dict.keys())

    # 4(i). separate the words from `all_selected_words` that are not in word2vec model
    uncommon_words = list()
    for i in all_selected_words:
        try:
            w2v_model[i]
        except KeyError as KE:
            uncommon_words.append(i)

    # 4(i(a)). if found write it to a text file
    if tech_terms_file:
        with open(tech_terms_file, 'a') as file:
            file.write('\n'.join(uncommon_words) + '\n')

    common_words = list(set(all_selected_words) - set(uncommon_words))

    # 4(i(b)). remove words with most similar meaning with each other
    unique_common_words = extract_unique_words(w2v_model, common_words)
    
    final_words_list = uncommon_words + unique_common_words
    final_dict = {}
    for w in final_words_list:
        final_dict[w] = int(word_counts_sorted_dict[w])
    return sort_dict_on_values(final_dict)

def parse_multiple_files(w2v_model: typing.Any, job_descriptions_dir: str) -> dict:
    '''
    A function that accepts all the '.txt' files inside ./job_descriptions/ directory
    Inside it, read all the files, gather all the words, extract the "important" words 
    and counts and returns a final sorted dict of word counts.

    Args:
        w2v_model - gensim word2vec model
        job_descriptions_dir - path to a directory containing job descriptions
    Returns:
        A sorted (based on values - counts) dict containing word-count
    '''
    text_files = sorted(glob(f"{job_descriptions_dir}/*"))
    imp_words = {}
    for file in text_files:
        # read a file
        contents = read_file(file)  
        terms = extract_word_counts(string=contents, w2v_model=w2v_model)
        for i in terms.keys():
            # if the word is in dict add the count 
            if i in imp_words:
                imp_words[i] = int(imp_words[i] + terms[i])
            # if not present add the word to the new dict
            else:
                imp_words[i] = int(terms[i])
    return sort_dict_on_values(imp_words)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        prog='JobAnalyser',
        description='This tool uses NLP with NLTK and Spacy to identify the required skills from a given job description. \
            Also does the same when multiple files are provided'
    )
    parser.add_argument(
        'path',
        type=str,
        help='Path to a single file or a directory containing multiple files to analyse'
    )
    parser.add_argument(
        '-s',
        '--single',
        dest='is_single',
        action="store_true",
        default=False,
        help='Whether the parsing is from a single file. \
            If false, the code operates for a directory instead'
    )
    parser.add_argument(
        '-w2v',
        '--word2vec-model-path',
        type=str,
        dest='word2vec_model_path',
        default='../GoogleNews-vectors-negative300.bin',
        help='path to the gensim word2vec model to use'
    )
    parser.add_argument(
        '-tt',
        '--tech-terms-file-path',
        type=str,
        dest='tech_terms_file_path',
        default='probably_tech_terms.txt',
        help='path to the file containing tech terms to pay attention to - one term / line'
    )
    args = parser.parse_args()

    print('Loading the required word2vec model')
    w2v_model = KeyedVectors.load_word2vec_format(
        args.word2vec_model_path,
        binary=True
    )

    if args.is_single:
       print(f'Parsing single file from {args.path}')
       contents_string = read_file(args.path)
       output = extract_word_counts(
            string=contents_string,
            w2v_model=w2v_model,
            tech_terms_file=args.tech_terms_file_path
        )
    else:
       print(f'Parsing multiple files from the directory {args.path}')
       output = parse_multiple_files(w2v_model=w2v_model, job_descriptions_dir=args.path)
    
    print('List of extracted terms...\n')
    print(json.dumps(output, indent=4))
    # print(pd.DataFrame([(k, v) for k, v in output.items()], columns=['Word', 'Count']))