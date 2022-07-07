def print_example(title, context, query, answer):
    """
    Print an example, constitued of a title, context,
    query and answer

    Parameters
    ----------
    title : str
    context : str
    query : str
    answer : dict

    Returns
    -------
    None
    """
    print(f"Title: {title}\n")
    print(f"Context: {context}\n")
    print(f"Query: {query}\n")
    print(f"Answer {answer}")