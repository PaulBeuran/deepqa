import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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


def tabularize_squad11_data(data):
    """
    Tabularize the SQuAD1.1 data dictionary into a list of examples,
    constitued of a tuple of subject title, a context, a query, and an answer.

    Parameters
    ----------
    data : dict
        SQuAD1.1 data dictionary

    Returns
    -------
    list
        A list  of examples, constitued of a tuple of subject title, 
        a context, a query, and an answer.
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
    return list(zip(titles, contexts, queries, answers))


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