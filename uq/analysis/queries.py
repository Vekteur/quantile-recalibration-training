def disjunction(queries):
    return ' or '.join(f'({query})' for query in queries)
