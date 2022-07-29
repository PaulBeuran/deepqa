import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from .tokenizer import BaseTokeniser
from .utils import char_ranges_to_token_ranges

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def unwrap_squad11_data(data):
    """
    Unwrap the SQuAD1.1 data dictionary into 4 lists containing respectively
    the subjects titles, the paragraphs contexts, the paragraphs queries and 
    the queries answers.

    Parameters
    ----------
    data : dict
        SQuAD1.1 data dictionary

    Returns
    -------
    list
        A list containing the subjects titles, the paragraphs contexts, 
        the paragraphs queries and the queries answers.
    """
    titles = list()
    contexts = list()
    queries = list()
    answers = list()
    for subject in data:
        title = subject["title"]
        titles.append(title)
        paragraphs = subject["paragraphs"]
        for paragraph in paragraphs:
            context = paragraph["context"]
            contexts.append(context)
            qas = paragraph["qas"]
            for qa in qas:
                question = qa["question"]
                answer = qa["answers"][0]["text"]
                queries.append(question)
                answers.append(answer)
    return titles, contexts, queries, answers


def tabularize_squad11_data(data, row_wise_output=False):
    """
    Tabularize the SQuAD1.1 data dictionary into a list of examples,
    constitued of four equally-sized unwrapped lists by default, else 
    a list of tuples of subject title, a context, a query, and an answer
    if a row-wise output is prefered.

    Parameters
    ----------
    data : dict
        SQuAD1.1 data dictionary
    row_wise_output, default=False
        Specify the output type, either column-wise by default (a tuple of equally-sized
        lists of titles, contexts, queries and answers respectively), or row-wise (a list of
        tuples containing each a title, a context, a query and an answer).

    Returns
    -------
    tuple or list
        A tuple of titles, contexts, questions and answers or
        a list of tuples containing each a title, a context, a query and an answer
    """
    titles = list()
    contexts = list()
    queries = list()
    answers = list()
    for subject in data:
        title = subject["title"]
        paragraphs = subject["paragraphs"]
        for paragraph in paragraphs:
            context = paragraph["context"]
            qas = paragraph["qas"]
            for qa in qas:
                question = qa["question"]
                answer = qa["answers"][0]
                titles.append(title)
                queries.append(question)
                contexts.append(context)
                answers.append(answer)
    if row_wise_output:
        return list(zip(titles, contexts, queries, answers))
    else:
        return titles, contexts, queries, answers


def get_corpus_tf_idf_word_frequencies(corpus, **tf_idf_vectorizer_kwargs):
    """
    Get the TF-IDF frequency for each words (or terms) in the corpus and return it
    as a dictionary. The TF-IDF computation is done using sklearn's
    TfidfVectorizer class, and as such can be controlled with its given
    parameters (ex: tokenizer, stop words, n-grams etc...)
    (see TfidfVectorizer documenation) 

    Parameters
    ----------
    corpus : list
        List of text or document corpus
    **tf_idf_vectorizer_kwargs : 
        Parameters given to the TfidfVectorizer class

    Returns
    -------
    dict
        A dictionary containing vocabulary terms associated with their respective
        TF-IDF frequency from the corpus
    """
    tf_idf_vectorizer = TfidfVectorizer(**tf_idf_vectorizer_kwargs)
    tf_idf = tf_idf_vectorizer.fit_transform(corpus)
    tf_idf_by_terms = np.asarray(tf_idf.sum(axis=0)).flatten()
    terms_tf_idf = (
        dict(
            zip(
                list(tf_idf_vectorizer.vocabulary_.keys()),
                tf_idf_by_terms[list(tf_idf_vectorizer.vocabulary_.values())]
            )
        )
    )
    return terms_tf_idf

def tokenize_squad_11_data(data, tokenizer, context_max_length, query_max_length):

    _, contexts, queries, answers = tabularize_squad11_data(data)
    contexts_token_ids, contexts_offset_mappings = tokenizer(contexts, context_max_length)
    queries_token_ids, _ = tokenizer(queries, max_length=query_max_length)
    answers_char_ranges = [(answer["answer_start"], answer["answer_start"] + len(answer["text"]))
                           for answer in answers]
    answers_token_ranges = char_ranges_to_token_ranges(
        answers_char_ranges, contexts_offset_mappings, context_max_length
    )
    return contexts_token_ids, queries_token_ids, answers_token_ranges

class QADataset(torch.utils.data.Dataset):
    def __init__(self, contexts, queries, answers):
        self.contexts = contexts
        self.queries = queries
        self.answers = answers

    def __getitem__(self, index):
        return (self.contexts[index], 
                self.queries[index], 
                self.answers[index])

    def __len__(self):
        return len(self.contexts)

