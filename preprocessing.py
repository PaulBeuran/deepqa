
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