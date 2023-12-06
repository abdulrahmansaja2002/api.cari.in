import random
from letor import Letor
from bsbi import BSBIIndex
from pathlib import Path
import sys
import os


FILE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))
def test():
    bsbi_instance = BSBIIndex.defaultInstance()
    letor = Letor()
    letor.load_model(as_ranker=True)

    # load query for testing
    queries = {} # query_id -> query
    queries_file = FILE_PATH / os.path.join("qrels-folder", "test_queries.txt")
    with open(queries_file) as f:
        for line in f:
            query_id, *query = line.strip().split(" ")
            queries[query_id] = query
    
    # load ground truth
    ground_truth = []
    ground_truth_file = FILE_PATH / os.path.join("qrels-folder", "test_qrels.txt")
    with open(ground_truth_file) as f:
        for line in f:
            query_id,doc_id = line.strip().split()
            ground_truth.append((query_id, doc_id))
    
    # choose random query
    query_id = random.choice(list(queries.keys()))
    query = queries[query_id]

    # load 100 docs for each query
    k = 100
    top_100_bm25 = bsbi_instance.retrieve_bm25(" ".join(query), k=k)
    top_100_doc_file = [doc_file for _, doc_file in top_100_bm25]
    top_100_docs_text = []
    for doc_file in top_100_doc_file:
        with open(doc_file) as f:
            top_100_docs_text.append(f.read())
        

    # re-rank
    top_100_letor = letor.re_ranking(" ".join(query),
                                     [f"{doc_id} {doc_text}" for doc_id, doc_text in enumerate(top_100_docs_text)])

    # calculate precision
    # precision = 0
    # for doc_id, _ in top_100_letor:
    #     if (query_id, doc_id) in ground_truth:
    #         precision += 1
    # precision /= k

    # print(f"Precision for query {query_id}: {precision}")

    # show comparison between bm25 and letor
    print("BM25\
          \n\tDoc ID\tScore")
    for id, (score, _) in enumerate(top_100_bm25):
        print(f"\t{id}\t{score:.2f}")

    print("Letor\
            \n\tDoc ID\tScore")
    for (did, score) in top_100_letor:
        print(f"\t{did}\t{score:.2f}")
    print()
    print("\tid \t doc_file")
    for id, (_, text_file) in enumerate(top_100_bm25):
        print(f"\t{id} \t {text_file}")

def init_index():
    bsbi_instance = BSBIIndex.defaultInstance()
    bsbi_instance.do_indexing()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "init_index": init_index()
    test()
    





    
