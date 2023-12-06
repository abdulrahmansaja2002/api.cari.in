from bsbi import BSBIIndex
from compression import VBEPostings

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir='collections',
                          postings_encoding=VBEPostings,
                          output_dir='index')

queries = ["Can the trophy system protect me against bullets?",
           "A problem about function N"]
for query in queries:
    print("Query  : ", query)
    print("Results:")
    for (score, doc, id) in BSBI_instance.retrieve_bm25(query, k=10):
        print(f"{id} {doc:30} {score:>.3f}")
        print(f"{BSBI_instance.doc_id_to_path(id)}")
    print()
