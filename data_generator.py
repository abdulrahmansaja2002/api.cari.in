import os
os.environ["IR_DATASETS_HOME"] = "D:/ir-datasets"

from pathlib import Path
import tqdm

import ir_datasets
gamming_dataset = ir_datasets.load("beir/cqadupstack/gaming")
math_dataset = ir_datasets.load("beir/cqadupstack/mathematica")
physics_dataset = ir_datasets.load("beir/cqadupstack/physics")
programmers_dataset = ir_datasets.load("beir/cqadupstack/programmers")

datasets = {
  "gaming": gamming_dataset,
  "math": math_dataset,
  "physics": physics_dataset,
  "programmers": programmers_dataset
}

num_per_docs = 20000

num_docs = 0
num_queries = 0
num_qrels = 0

collections_path = Path("collections")
qrels_path = Path("qrels")
queries_path = Path("queries")

def init_collections(datasets=datasets, num_per_docs=num_per_docs, collections_path=collections_path):
  for name, dataset in datasets.items():
    path = collections_path / name
    path.mkdir(parents=True, exist_ok=True)

    for doc in tqdm.tqdm(dataset.docs_iter()[:num_per_docs], desc=f"Creating {name} docs"):
      with open(path / f"{doc.doc_id}.txt", "w", encoding="utf-8") as f:
        title = doc.title
        text = doc.text
        tags = doc.tags
        f.write(f"{title}\n{text}\n{tags}\n")


def init_qrels(datasets=datasets, qrels_path=qrels_path, collections_path=collections_path):  
  for name, dataset in datasets.items():
    for qrels in tqdm.tqdm(dataset.qrels_iter(), desc=f"Creating {name} qrels"):
      with open(qrels_path / f"{name}.txt", "a", encoding="utf-8") as f:
        doc_id = qrels.doc_id
        if f"{doc_id}.txt" in os.listdir(collections_path / name):
          f.write(f"{qrels.query_id} {qrels.iteration} {doc_id} {qrels.relevance}\n")

def init_queries(datasets=datasets, queries_path=queries_path):
  for name, dataset in datasets.items():
    path = queries_path 

    for query in tqdm.tqdm(dataset.queries_iter(), desc=f"Creating {name} queries"):
      with open(path / f"{name}.txt", "a", encoding="utf-8") as f:
        query_id = query.query_id
        text = query.text
        f.write(f"{query_id} {text}\n")


def showOneExample():
  for name, dataset in datasets.items():
    print(name)
    for doc in dataset.docs_iter():
      print(doc)
      break
    for query in dataset.queries_iter():
      print(query)
      break
    for qrels in dataset.qrels_iter():
      print(qrels)
      break
    break
  


if __name__ == "__main__":
  init_collections()
  init_qrels()
  init_queries()
