import urllib.parse as uparse
import urllib.request as urequest

def ids_converter(queries, p_from='ACC', p_to='ID'):
    url = "https://www.uniprot.org/uploadlists/"
    params = {"from": p_from, "to": p_to, "format": "tab", "query": "\t".join(queries)}
    data = uparse.urlencode(params)
    data = data.encode("utf-8")
    req = urequest.Request(url, data)
    with urequest.urlopen(req) as f:
        response = f.read()
    results = [line.split('\t') for line in response.decode("utf-8").split('\n')[1:-1]]
    results = {line[0]: line[1] for line in results}
    results = [results.get(code, None) for code in queries]
    print(len(queries), len(results))
    return results