from collections import defaultdict
import os
import pickle
import random
import tqdm

class IdMap:
    """
    Ingat kembali di kuliah, bahwa secara praktis, sebuah dokumen dan
    sebuah term akan direpresentasikan sebagai sebuah integer. Oleh
    karena itu, kita perlu maintain mapping antara string term (atau
    dokumen) ke integer yang bersesuaian, dan sebaliknya. Kelas IdMap ini
    akan melakukan hal tersebut.
    """

    def __init__(self):
        """
        Mapping dari string (term atau nama dokumen) ke id disimpan dalam
        python's dictionary; cukup efisien. Mapping sebaliknya disimpan dalam
        python's list.

        contoh:
            str_to_id["halo"] ---> 8
            str_to_id["/collection/dir0/gamma.txt"] ---> 54

            id_to_str[8] ---> "halo"
            id_to_str[54] ---> "/collection/dir0/gamma.txt"
        """
        self.str_to_id = {}
        self.id_to_str = []

    def __len__(self):
        """Mengembalikan banyaknya term (atau dokumen) yang disimpan di IdMap."""
        # TODO
        return len(self.id_to_str)

    def __get_id(self, s):
        """
        Mengembalikan integer id i yang berkorespondensi dengan sebuah string s.
        Jika s tidak ada pada IdMap, lalu assign sebuah integer id baru dan kembalikan
        integer id baru tersebut.
        """
        # TODO
        if s not in self.str_to_id:
            self.str_to_id[s] = len(self.id_to_str)
            self.id_to_str.append(s)
        
        return self.str_to_id[s]

    def __get_str(self, i):
        """Mengembalikan string yang terasosiasi dengan index i."""
        # TODO
        return self.id_to_str[i]

    def __getitem__(self, key):
        """
        __getitem__(...) adalah special method di Python, yang mengizinkan sebuah
        collection class (seperti IdMap ini) mempunyai mekanisme akses atau
        modifikasi elemen dengan syntax [..] seperti pada list dan dictionary di Python.

        Silakan search informasi ini di Web search engine favorit Anda. Saya mendapatkan
        link berikut:

        https://stackoverflow.com/questions/43627405/understanding-getitem-method

        Jika key adalah integer, gunakan __get_str;
        jika key adalah string, gunakan __get_id
        """
        # TODO
        return self.__get_id(key) if isinstance(key, str) else self.__get_str(key)


def merge_and_sort_posts_and_tfs(posts_tfs1, posts_tfs2):
    """
    Menggabung (merge) dua lists of tuples (doc id, tf) dan mengembalikan
    hasil penggabungan keduanya (TF perlu diakumulasikan untuk semua tuple
    dengn doc id yang sama), dengan aturan berikut:

    contoh: posts_tfs1 = [(1, 34), (3, 2), (4, 23)]
            posts_tfs2 = [(1, 11), (2, 4), (4, 3 ), (6, 13)]

            return   [(1, 34+11), (2, 4), (3, 2), (4, 23+3), (6, 13)]
                   = [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)]

    Parameters
    ----------
    list1: List[(Comparable, int)]
    list2: List[(Comparable, int]
        Dua buah sorted list of tuples yang akan di-merge.

    Returns
    -------
    List[(Comparable, int)]
        Penggabungan yang sudah terurut
    """
    # TODO
    def compare(postA, postB) :
        return postA[0] - postB[0]
    def flush(dst, target):
        return target + dst
    result = []
    i_a, i_b, n_a, n_b, = 0, 0, len(posts_tfs1), len(posts_tfs2)
    while i_a < n_a and i_b < n_b:
        if compare(posts_tfs1[i_a], posts_tfs2[i_b]) == 0:
            new_value = (posts_tfs1[i_a][0], posts_tfs1[i_a][1] + posts_tfs2[i_b][1])
            result.append(new_value)
            i_a += 1
            i_b += 1
        elif compare(posts_tfs1[i_a],  posts_tfs2[i_b]) < 0:
            result.append(posts_tfs1[i_a])
            i_a += 1
        else:
            result.append(posts_tfs2[i_b])
            i_b += 1
    result = flush(posts_tfs1[i_a:], result)
    result = flush(posts_tfs2[i_b:], result)
    return result


# taken from : https://www.geeksforgeeks.org/python-text-summarizer/
# importing libraries 
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
def summarize_text(text):
    # Tokenizing the text 
    stopWords = set(stopwords.words("indonesian")) 
    words = word_tokenize(text) 
      
    # Creating a frequency table to keep the  
    # score of each word 
      
    freqTable = dict() 
    for word in words: 
        word = word.lower() 
        if word in stopWords: 
            continue
        if word in freqTable: 
            freqTable[word] += 1
        else: 
            freqTable[word] = 1
      
    # Creating a dictionary to keep the score 
    # of each sentence 
    sentences = sent_tokenize(text) 
    sentenceValue = dict() 
      
    for sentence in sentences: 
        for word, freq in freqTable.items(): 
            if word in sentence.lower(): 
                if sentence in sentenceValue: 
                    sentenceValue[sentence] += freq 
                else: 
                    sentenceValue[sentence] = freq 
      
      
      
    sumValues = 0
    for sentence in sentenceValue: 
        sumValues += sentenceValue[sentence] 
      
    # Average value of a sentence from the original text 
      
    average = int(sumValues / len(sentenceValue)) 
      
    # Storing sentences into our summary. 
    summary = '' 
    for sentence in sentences: 
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)): 
            summary += " " + sentence 
    return summary
    


def load_queries(queries_file="test_queries.txt", qids=[]):
    queries = {}
    with open(queries_file, encoding="utf-8") as file:
        for line in file:
            q_id, *content = line.split()
            q_id = int(q_id)
            if q_id in qids:
                queries[int(q_id)] = content
    return queries


def load_qrels(qrel_file="test_qrels.txt", train_docs_ids=[]):
    qrels = defaultdict(lambda: defaultdict(lambda: 0))
    qids = set()
    with open(qrel_file, encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            qid = int(parts[0])
            did = int(parts[2])
            rel = int(parts[3])
            if did in train_docs_ids:
                qrels[qid][did] = rel
                qids.add(qid)
    return qrels, qids

def get_doc_sep():
    with open(os.path.join("index", "doc.sep"), "rb") as file:
        return pickle.load(file)

def load_docs(doc_paths=["collections/math/1.txt", "collections/math/2.txt"]):
    docs = {}
    doc_sep = get_doc_sep()
    for docs_file in tqdm.tqdm(doc_paths, desc=f"Loading docs {'/'.join(doc_paths[0].split(doc_sep)[:-2])}/*.txt"):
        with open(docs_file, encoding="utf-8") as file:
            title, text, tags = file.readlines()
            title, text, tags = title.strip(), text.strip(), tags.strip()
            id = docs_file.split(doc_sep)[-1].split(".")[0]
            content = f"{title} {text}".split()
            docs[int(id)] = content
    return docs


def load_data(docs_path, queries_file, q_docs_rel_file, num_negatives, train_size=0.75):
    train_docs = os.listdir(docs_path)[:int(len(os.listdir(docs_path)) * train_size)]
    doc_paths = [os.path.join(docs_path, doc) for doc in train_docs]
    documents = load_docs(doc_paths)
    train_docs_ids = [int(doc.split(get_doc_sep())[-1].split(".")[0]) for doc in doc_paths]
    q_docs_rel, qids = load_qrels(q_docs_rel_file, train_docs_ids)
    queries = load_queries(queries_file, qids)

    # group_qid_count untuk model LGBMRanker
    group_qid_count = []
    dataset = []
    for q_id in q_docs_rel:
        docs_rels = q_docs_rel[q_id]
        group_qid_count.append(len(docs_rels) + num_negatives)
        for doc_id, rel in docs_rels.items():
            dataset.append((queries[q_id], documents[doc_id], rel))
        # tambahkan satu negative (random sampling saja dari documents)
        dataset.append((queries[q_id], random.choice(list(documents.values())), 0))
    return documents, dataset, group_qid_count


if __name__ == '__main__':

    doc = ["halo", "semua", "selamat", "pagi", "semua"]
    term_id_map = IdMap()

    assert [term_id_map[term]
            for term in doc] == [0, 1, 2, 3, 1], "term_id salah"
    assert term_id_map[1] == "semua", "term_id salah"
    assert term_id_map[0] == "halo", "term_id salah"
    assert term_id_map["selamat"] == 2, "term_id salah"
    assert term_id_map["pagi"] == 3, "term_id salah"

    docs = ["/collection/0/data0.txt",
            "/collection/0/data10.txt",
            "/collection/1/data53.txt"]
    doc_id_map = IdMap()
    assert [doc_id_map[docname]
            for docname in docs] == [0, 1, 2], "docs_id salah"

    assert merge_and_sort_posts_and_tfs([(1, 34), (3, 2), (4, 23)],
                                        [(1, 11), (2, 4), (4, 3), (6, 13)]) == [(1, 45), (2, 4), (3, 2), (4, 26), (6, 13)], "merge_and_sort_posts_and_tfs salah"
