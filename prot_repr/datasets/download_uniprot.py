import os
import click
import requests
from prot_repr.utils import dataset_folder


@click.command(help="Programmatic access - Downloading data at every UniProt release")
@click.option('--organism-taxon', type=str, default='human',
              help="Unique name for the organism which proteins you want to download")
def main(organism_taxon):
    # Last - Modified
    query = "https://www.uniprot.org/uniprot/?query=organism:{}&format=fasta".format(organism_taxon)
    file = os.path.join(dataset_folder, f'uniprotkb_{organism_taxon}.fasta')

    response = requests.get(query)
    if response.status_code != 200:
        # This means something went wrong.
        raise Exception('Something went wrong with status {}'.format(response.status_code))
    with open(file, 'w') as fd:
        fd.write(response.content.decode("utf-8"))

    results = response.headers['X-Total-Results']
    release = response.headers['X-UniProt-Release']
    date = response.headers['Last-Modified']
    print(f"Downloaded {results} entries of UniProt release {release} ({date}) to file {file}\n")

if __name__ == '__main__':
    main()



b"abcde".decode("utf-8")