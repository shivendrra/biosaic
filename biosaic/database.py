import os, time
from Bio import Entrez, SeqIO

def search_ncbi(query, db='nucleotide', retmax=10000, email=None, api_key=None):
  # configuring Entrez
  if email:
    Entrez.email = email
  if api_key: # optional (when exceeding the rate limits)
    Entrez.api_key = api_key

  # ESearch to get list of UIDs
  handle = Entrez.esearch(db=db, term=query, retmax=retmax)
  record = Entrez.read(handle)
  handle.close()
  return record.get('IdList', [])

def fetch_and_save(ids, db='nucleotide', out_dir='sequences', batch_size=500, format='fasta'):
  os.makedirs(out_dir, exist_ok=True)
  for start in range(0, len(ids), batch_size):
    batch_ids = ids[start:start+batch_size]

    handle = Entrez.efetch(db=db, id=','.join(batch_ids), rettype=format, retmode='text')
    records = SeqIO.parse(handle, format)
    out_path = os.path.join(out_dir, f'sequences_{start+1}_{start+len(batch_ids)}.fasta')

    SeqIO.write(records, out_path, format)
    handle.close()

    print(f'Saved {len(batch_ids)} records to {out_path}')
    time.sleep(0.4)  # NCBI recommends <= 3 requests/sec

def get_database(query, output_dir, db="nucleotide", retmax=10000, email=None, api_key=None, batch_size=500):
  """ handles calling the fetch_save()
    Args: 
      query (str): Entrez query (e.g., "Homo sapiens[Organism] AND COX1[Gene]")
      output_dir (str, path): Directory to save FASTA files
      db (str, optional): NCBI database to search. Defaults to "nucleotide".
      retmax (int, optional): Max number of records to retrieve. Defaults to 10000.
      email (str, required): Your email (required by NCBI). replace the None.
      api_key (str, optional): NCBI API key for higher rate limits. Defaults to None.
      batch_size (int, optional): Number of sequences per file. Defaults to 500.
  """
  ids = search_ncbi(query, db=db, retmax=retmax, email=email, api_key=api_key)  # :contentReference[oaicite:0]{index=0}
  print(f'Found {len(ids)} sequence IDs for query: {query}')
  fetch_and_save(ids, db=db, out_dir=output_dir, batch_size=batch_size)  # :contentReference[oaicite:1]{index=1}