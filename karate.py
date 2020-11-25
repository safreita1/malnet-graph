from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from karateclub.utils.treefeatures import WeisfeilerLehmanHashing
from karateclub.graph_embedding import GeoScattering, FeatherGraph, IGE, Graph2Vec, GL2Vec, NetLSD, FGSD, SF

# too slow: fgsd, ige, GeoScattering, netlsd, sf
# too much space: g2v, gl2vec


def feather(graph):
    model = FeatherGraph()
    model._set_seed()
    model._check_graphs([graph])

    embedding = model._calculate_feather(graph)
    return embedding


def ige(graph, max_deg):
    model = IGE()
    model._set_seed()
    model._check_graphs([graph])
    model.max_deg = max_deg

    embedding = model._calculate_invariant_embedding(graph)
    return embedding


def fgsd(graph):
    model = FGSD()
    model._set_seed()
    model._check_graphs([graph])

    embedding = model._calculate_fgsd(graph)
    return embedding


def lsd(graph):
    model = NetLSD()
    model._set_seed()
    model._check_graphs([graph])

    embedding = model._calculate_netlsd(graph)
    return embedding


def sf(graph):
    model = SF()
    model._set_seed()
    model._check_graphs([graph])

    embedding = model._calculate_sf(graph)
    return embedding


def geo_scattering(graph):
    model = GeoScattering()
    model._set_seed()
    model._check_graphs([graph])

    embedding = model._calculate_geoscattering(graph)
    return embedding


def g2v(idx, graph):
    model = Graph2Vec()
    model._set_seed()
    model._check_graphs([graph])

    document = WeisfeilerLehmanHashing(graph, model.wl_iterations, model.attributed, model.erase_base_features)
    document = TaggedDocument(words=document.get_graph_features(), tags=str(idx))
    return document


def g2v_embed(documents):
    model = Doc2Vec(documents)

    embedding = [model.docvecs[str(i)] for i, _ in enumerate(documents)]
    return embedding
