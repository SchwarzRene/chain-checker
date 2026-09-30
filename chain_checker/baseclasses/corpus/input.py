from chain_checker.baseclasses.corpus.record import Record


class Input(Record):
    # A dedicated subclass rather than using Record directly, so an input can
    # be type-checked against Input specifically (e.g. Model.__call__ rejects
    # anything that isn't one) even though Label and ModelOutput are also
    # Records.
    pass
