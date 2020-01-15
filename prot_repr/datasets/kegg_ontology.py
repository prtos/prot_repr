from urllib.error import HTTPError
from Bio.KEGG.REST import kegg_get
from tqdm import trange
from prot_repr.datasets.ids_conversion import ids_converter


def get_kegg_ontology(koids):
    r"""
    Get the ontology of proteins

    Arguments
    ----------
        geneid: str
            KEGG gene id, for the gene of interest

    Returns
    -------
        pathways:
            List of pathways (Bio.KEGG.KGML.KGML_pathway.Pathway) in which the input drug is involved
    """
    step = 10
    all_res = {}
    for i in trange(0, len(koids), step):
        try:
            res = kegg_get(koids[i:i+step]).read()
        except HTTPError:
            continue
        for koid, entry in zip(koids[i:i+step], res.split('ENTRY')[1:]):
            stop_text = 'DBLINKS' if 'DBLINKS' in entry else 'GENES'
            start, stop = entry.find('BRITE'), entry.find(stop_text)
            all_res[koid] = entry[start:stop]

    return [all_res.get(code, None) for code in koids]

