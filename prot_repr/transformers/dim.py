from prot_repr.models.trainer import load_model, FastaDataset

def dim_transform(proteins):
    model = load_model()
    x = FastaDataset.transform(proteins)
    return model.network.encode(x)