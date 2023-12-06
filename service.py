from bsbi import BSBIIndex
from compression import VBEPostings
import time
import os
from letor import Letor

BSBI_instance = None
index_folder = os.environ.get('INDEX_FOLDER', 'index')
num_results = int(os.environ.get('NUM_RESULTS', 100))
letor = None
def get_indexing_instance():
  global BSBI_instance
  if BSBI_instance is None:
    BSBI_instance = BSBIIndex(data_dir='collections',
                            postings_encoding=VBEPostings,
                            output_dir=index_folder)
  return BSBI_instance

def get_by_id(id):
  BSBI_instance = get_indexing_instance()
  doc_path = BSBI_instance.doc_id_to_path(id)
  doc_sep = BSBI_instance.get_doc_sep()
  norm_path = os.path.normpath(doc_path)
  norm_path = norm_path.replace(doc_sep, os.path.sep)
  doc_type = norm_path.split(os.path.sep)[1]
  with open(norm_path, "r", encoding="utf-8") as f:
    title, text, tags = f.readlines()
    return {
      "id": id,
      "title": title.strip(),
      "text": text.strip(),
      "tags": eval(tags.strip()),
      "type": doc_type
    }

def rerank(docs, query):
  global letor
  if letor is None:
    letor = Letor(init_data=False)
    letor.load_model(as_ranker=True)

  list_docs = []
  for (_, doc_path, id) in docs:
    doc_sep = BSBI_instance.get_doc_sep()
    norm_path = os.path.normpath(doc_path)
    norm_path = norm_path.replace(doc_sep, os.path.sep)
    with open(norm_path, "r", encoding="utf-8") as f:
      title, text, _ = f.readlines()
      list_docs.append(f"{id} {title.strip()} {text.strip()}")
  
  return letor.re_ranking(query, list_docs)
  


def search(query):
  BSBI_instance = get_indexing_instance()
  docs = []
  start = time.time()
  docs_results = BSBI_instance.retrieve_bm25(query, k=num_results)

  # reranking
  useLetor = eval(os.environ.get('USE_LETOR', "False"))
  if useLetor:
    docs_results = rerank(docs_results, query)
    end = time.time()
    if docs_results is None:
      return None
    else:
      for (doc_id, score) in docs_results:
        doc_data = get_by_id(int(doc_id))
        doc_data["score"] = score
        docs.append(doc_data)
  else:
    end = time.time()
    doc_sep = BSBI_instance.get_doc_sep()
    for (score, doc_path, id) in docs_results:
      norm_path = os.path.normpath(doc_path)
      norm_path = norm_path.replace(doc_sep, os.path.sep)
      doc_type = norm_path.split(os.path.sep)[1]
      with open(norm_path, "r", encoding="utf-8") as f:
        title, text, tags = f.readlines()
        docs.append({
          "id": id,
          "title": title.strip(),
          "text": text.strip(),
          "tags": eval(tags.strip()),
          "score": score,
          "type": doc_type
        })
  result = {
    "total": len(docs),
    "docs": docs,
    "time": end - start
  }
  return result
    