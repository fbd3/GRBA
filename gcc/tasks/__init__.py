from gcc.models.emb import (
    FromNumpyGraph,
)


def build_model(name, hidden_size, **model_args):
    return {
        "from_numpy_graph": FromNumpyGraph,
    }[name](hidden_size, **model_args)
