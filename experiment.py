import math
import re
import os
import time

from pathlib import Path
from bsbi import BSBIIndex
from compression import VBEPostings
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from letor import Letor

# >>>>> 3 IR metrics: RBP p = 0.8, DCG, dan AP
FILE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

def rbp(ranking, p=0.8):
    """ menghitung search effectiveness metric score dengan 
        Rank Biased Precision (RBP)

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score RBP
    """
    score = 0.
    for i in range(1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * (p ** (i - 1))
    return (1 - p) * score


def dcg(ranking):
    """ menghitung search effectiveness metric score dengan 
        Discounted Cumulative Gain

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score DCG
    """
    # TODO
    score = 0
    for i in range (1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] / math.log2(i+1)
    return score


def prec(ranking, k):
    """ menghitung search effectiveness metric score dengan 
        Precision at K

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        k: int
          banyak dokumen yang dipertimbangkan atau diperoleh

        Returns
        -------
        Float
          score Prec@K
    """
    # TODO
    score = 0
    for i in range (k):
        score += ranking[i] / k
    return score


def ap(ranking):
    """ menghitung search effectiveness metric score dengan 
        Average Precision

        Parameters
        ----------
        ranking: List[int]
           vektor biner seperti [1, 0, 1, 1, 1, 0]
           gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
           Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                   di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                   di rank-6 tidak relevan

        Returns
        -------
        Float
          score AP
    """
    # TODO
    score = 0
    R = sum(ranking)
    for i in range (1, len(ranking) + 1):
        pos = i - 1
        score += ranking[pos] * prec(ranking, i) / R
    return score

# >>>>> memuat qrels


def load_qrels(qrel_file="qrels.txt"):
    """ 
        memuat query relevance judgment (qrels) 
        dalam format dictionary of dictionary qrels[query id][document id],
        dimana hanya dokumen yang relevan (nilai 1) yang disimpan,
        sementara dokumen yang tidak relevan (nilai 0) tidak perlu disimpan,
        misal {"Q1": {500:1, 502:1}, "Q2": {150:1}}
    """
    with open(qrel_file) as file:
        content = file.readlines()

    qrels_sparse = {}

    for line in content:
        parts = line.strip().split()
        qid = parts[0]
        did = int(parts[1])
        if not (qid in qrels_sparse):
            qrels_sparse[qid] = {}
        if not (did in qrels_sparse[qid]):
            qrels_sparse[qid][did] = 0
        qrels_sparse[qid][did] = 1
    return qrels_sparse

# >>>>> EVALUASI !


def eval_retrieval(qrels, query_file="queries.txt", k=1000):
    """ 
      loop ke semua query, hitung score di setiap query,
      lalu hitung MEAN SCORE-nya.
      untuk setiap query, kembalikan top-1000 documents
    """
    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    num_queries = 0
    with open(query_file) as file:
        rbp_scores_bm25 = []
        dcg_scores_bm25 = []
        ap_scores_bm25 = []
        rbp_scores_letor = []
        dcg_scores_letor = []
        ap_scores_letor = []
        rbp_scores_tfidf = []
        dcg_scores_tfidf = []
        ap_scores_tfidf = []
        rbp_scores_letor_tfidf = []
        dcg_scores_letor_tfidf = []
        ap_scores_letor_tfidf = []



        letor = Letor()
        letor.load_model(as_ranker=True)

        for qline in tqdm(file):
            num_queries += 1
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])
            

            """
            Evaluasi BM25
            """
            ranking_bm25 = []
            # nilai k1 dan b dapat diganti-ganti
            bm25 = BSBI_instance.retrieve_bm25(query, k=k)
            if len(bm25) == 0:
                continue
            for (score, doc) in bm25:
                did = int(os.path.splitext(os.path.basename(doc))[0])
                # Alternatif lain:
                # 1. did = int(doc.split("\\")[-1].split(".")[0])
                # 2. did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
                # 3. disesuaikan dengan path Anda
                if (did in qrels[qid]):
                    ranking_bm25.append(1)
                else:
                    ranking_bm25.append(0)
            rbp_scores_bm25.append(rbp(ranking_bm25))
            dcg_scores_bm25.append(dcg(ranking_bm25))
            ap_scores_bm25.append(ap(ranking_bm25))

            """
            Evaluasi Re-ranking dengan LGBMRanker
            """
            ranking_letor = []

            # re-ranking dengan LGBMRanker
            docs = [os.path.join("collections", doc) for (_, doc) in bm25]
            docs_text = []
            for doc in docs:
                with open(doc, encoding='utf-8') as file:
                    text = file.read()
                    did = int(os.path.splitext(os.path.basename(doc))[0])
                    docs_text.append(f"{did} {text}")
            for (did, score) in letor.re_ranking(query, docs_text):
                if (int(did) in qrels[qid]):
                    ranking_letor.append(1)
                else:
                    ranking_letor.append(0)

            rbp_scores_letor.append(rbp(ranking_letor))
            dcg_scores_letor.append(dcg(ranking_letor))
            ap_scores_letor.append(ap(ranking_letor)) 

            """
            Evaluasi TF-IDF
            """
            ranking_tfidf = []
            tfidf = BSBI_instance.retrieve_tfidf(query, k=k)
            if len(tfidf) == 0:
                continue
            for (score, doc) in tfidf:
                did = int(os.path.splitext(os.path.basename(doc))[0])
                if (did in qrels[qid]):
                    ranking_tfidf.append(1)
                else:
                    ranking_tfidf.append(0)
            rbp_scores_tfidf.append(rbp(ranking_tfidf))
            dcg_scores_tfidf.append(dcg(ranking_tfidf))
            ap_scores_tfidf.append(ap(ranking_tfidf))

            """
            Evaluasi Re-ranking dengan LGBMRanker dan TF-IDF
            """
            docs = [os.path.join("collections", doc) for (_, doc) in tfidf]
            docs_text = []
            for doc in docs:
                with open(doc, encoding='utf-8') as file:
                    text = file.read()
                    did = int(os.path.splitext(os.path.basename(doc))[0])
                    docs_text.append(f"{did} {text}")
            ranking_letor_tfidf = []
            for (did, score) in letor.re_ranking(query, docs_text):
                if (int(did) in qrels[qid]):
                    ranking_letor_tfidf.append(1)
                else:
                    ranking_letor_tfidf.append(0)
            
            rbp_scores_letor_tfidf.append(rbp(ranking_letor_tfidf))
            dcg_scores_letor_tfidf.append(dcg(ranking_letor_tfidf))
            ap_scores_letor_tfidf.append(ap(ranking_letor_tfidf))


        print(f"Hasil evaluasi BM25 terhadap {num_queries} queries")
        print("RBP score =", sum(rbp_scores_bm25) / len(rbp_scores_bm25))
        print("DCG score =", sum(dcg_scores_bm25) / len(dcg_scores_bm25))
        print("AP score  =", sum(ap_scores_bm25) / len(ap_scores_bm25))

        print(f"Hasil evaluasi Re-ranking dengan LGBMRanker terhadap {num_queries} queries")
        print("RBP score =", sum(rbp_scores_letor) / len(rbp_scores_letor))
        print("DCG score =", sum(dcg_scores_letor) / len(dcg_scores_letor))
        print("AP score  =", sum(ap_scores_letor) / len(ap_scores_letor))

        print(":"*50)

        print(f"Hasil evaluasi TF-IDF terhadap {num_queries} queries")
        print("RBP score =", sum(rbp_scores_tfidf) / len(rbp_scores_tfidf))
        print("DCG score =", sum(dcg_scores_tfidf) / len(dcg_scores_tfidf))
        print("AP score  =", sum(ap_scores_tfidf) / len(ap_scores_tfidf))

        print(f"Hasil evaluasi Re-ranking dengan LGBMRanker dan TF-IDF terhadap {num_queries} queries")
        print("RBP score =", sum(rbp_scores_letor_tfidf) / len(rbp_scores_letor_tfidf))
        print("DCG score =", sum(dcg_scores_letor_tfidf) / len(dcg_scores_letor_tfidf))
        print("AP score  =", sum(ap_scores_letor_tfidf) / len(ap_scores_letor_tfidf))


def k1_b_bm25_experiment(qrels):
    k1_values = [0.5, 1, 1.5, 2, 2.5, 3]
    b_values = [0.25, 0.5, 0.75, 1]

    BSBI_instance = BSBIIndex(data_dir='collections',
                                postings_encoding=VBEPostings,
                                output_dir='index')
    
    with open("queries.txt") as file:
        rbp_scores_bm25 = {}
        dcg_scores_bm25 = {}
        ap_scores_bm25 = {}
        score_formats = []
        is_new_score_format = True
        for qline in tqdm(file):
            parts = qline.strip().split()
            qid = parts[0]
            query = " ".join(parts[1:])


            for k1 in k1_values:
                for b in b_values:
                    ranking_bm25 = []
                    for (score, doc) in BSBI_instance.retrieve_bm25(query, k=1000, k1=k1, b=b):
                        did = int(os.path.splitext(os.path.basename(doc))[0])
                        if (did in qrels[qid]):
                            ranking_bm25.append(1)
                        else:
                            ranking_bm25.append(0)
                    score_format = f"k1 = {k1}, b = {b}"
                    rbp_scores_bm25[score_format] = rbp_scores_bm25.get(score_format, []) + [rbp(ranking_bm25)]
                    dcg_scores_bm25[score_format] = dcg_scores_bm25.get(score_format, [])+ [dcg(ranking_bm25)]
                    ap_scores_bm25[score_format] = ap_scores_bm25.get(score_format, []) + [ap(ranking_bm25)]
                    if is_new_score_format: score_formats.append(score_format)
            
            is_new_score_format = False

        # create plot
        fig, ax = plt.subplots()
        index = np.arange(len(score_formats))
        bar_width = 0.25
        opacity = 0.8

        rbp_means = [sum(rbp_scores_bm25[score_format]) / len(rbp_scores_bm25[score_format]) for score_format in score_formats]
        dcg_means = [sum(dcg_scores_bm25[score_format]) / len(dcg_scores_bm25[score_format]) for score_format in score_formats]
        ap_means = [sum(ap_scores_bm25[score_format]) / len(ap_scores_bm25[score_format]) for score_format in score_formats]

        # normalize
        rbp_means = [rbp_mean / max(rbp_means) for rbp_mean in rbp_means]
        dcg_means = [dcg_mean / max(dcg_means) for dcg_mean in dcg_means]
        ap_means = [ap_mean / max(ap_means) for ap_mean in ap_means]
        
        rbp_bar = plt.bar(index, rbp_means, bar_width,
        alpha=opacity,
        color='b',
        label='RBP')

        dcg_bar = plt.bar(index + bar_width, dcg_means, bar_width,
        alpha=opacity,
        color='g',
        label='DCG')

        ap_bar = plt.bar(index + 2*bar_width, ap_means, bar_width,
        alpha=opacity,
        color='r',
        label='AP')

        plt.xlabel('Parameter')
        plt.ylabel('Score')
        plt.title('Scores by parameter')
        plt.xticks(index + bar_width, score_formats)
        # Verticalize the labels
        plt.xticks(rotation=90)


        plt.legend()

        plt.tight_layout()
        plt.show()



        print("Hasil evaluasi BM25 terhadap 150 queries")
        for score_format in score_formats:
            print(f"{score_format}")
            print("RBP score =", sum(rbp_scores_bm25[score_format]) / len(rbp_scores_bm25[score_format]))
            print("DCG score =", sum(dcg_scores_bm25[score_format]) / len(dcg_scores_bm25[score_format]))
            print("AP score  =", sum(ap_scores_bm25[score_format]) / len(ap_scores_bm25[score_format]))


def compare_wand_algorithm():
    BSBI_instance = BSBIIndex(data_dir='collections',
                                postings_encoding=VBEPostings,
                                output_dir='index')
    
    # call this function to calculate upper bound score for each term
    # call this function only once, and comment it after that
    BSBI_instance.calculate_deafult_ubs()

    # if we want to experiment with k1 and b, uncomment this
    # k1 = [your value]
    # b = [your value]
    # with InvertedIndexReader("main_index", postings_encoding=VBEPostings, directory="index") as index:
    #    index.count_all_term_upper_bound_score(k1, b)

    queries = ["Jumlah uang terbatas yang telah ditentukan sebelumnya bahwa seseorang harus membayar dari tabungan mereka sendiri",
           "Terletak sangat dekat dengan khatulistiwa"]
    
    for query in queries:
        print(f"Query: {query}")
        print("WAND Algorithm with bm25 scoring")
        start = time.time()
        for (score, doc) in BSBI_instance.retrieve_wand(query, k=10, scoring='bm25'):
            print(f"{doc} {score}")
        end = time.time()
        print(f"Time: {end - start}")
        print("BM25 Algorithm")
        start = time.time()
        for (score, doc) in BSBI_instance.retrieve_bm25(query, k=10):
            print(f"{doc} {score}")
        end = time.time()
        print(f"Time: {end - start}")
        print("WAND Algorithm with tfidf scoring")
        start = time.time()
        for (score, doc) in BSBI_instance.retrieve_wand(query, k=10, scoring='tfidf'):
            print(f"{doc} {score}")
        end = time.time()
        print("TF-IDF Algorithm")
        start = time.time()
        for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=10):
            print(f"{doc} {score}")
        end = time.time()
        print(f"Time: {end - start}")
        print("")
            

if __name__ == '__main__':
    qrels = load_qrels(FILE_PATH / os.path.join("qrels-folder", "test_qrels.txt"))

    # assert qrels["Q1002252"][5599474] == 1, "qrels salah"
    # assert not (6998091 in qrels["Q1007972"]), "qrels salah"

    eval_retrieval(
        qrels,
        FILE_PATH / os.path.join("qrels-folder", "test_queries.txt"),
        k=100
    )
    # k1_b_bm25_experiment(qrels)
    # compare_wand_algorithm()
