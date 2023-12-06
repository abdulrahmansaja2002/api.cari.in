import os
import pickle
import contextlib
import heapq
import math
import re

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, merge_and_sort_posts_and_tfs
from compression import VBEPostings
from tqdm import tqdm

from nltk import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, SnowballStemmer
# from mpstemmer import MPStemmer
# from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

from operator import itemgetter
from collections import Counter
import sys

from wand import WAND, calculate_bm25, calculate_tfidf

import nltk
from nltk.corpus import stopwords as nltk_stopwords
try: 
    nltk.download('stopwords')
    nltk.download('punkt')
except:
    pass

DEBUG = False
class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

        # stopwords
        self.stopwords = set(nltk_stopwords.words('english'))

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
    def remove_stopwords(self, content):
        return [word for word in content if word.lower() not in self.stopwords]
    def pre_processing_text(self, content):
        """
        Melakukan preprocessing pada text, yakni stemming dan removing stopwords
        """
        # get sentences from text
        # stemmer = PorterStemmer()
        stemmer = SnowballStemmer("english")
        # pattern = r"\w+(?:'\w+)?|[^\w\s]"
        # tokens = re.findall(pattern, content)
        # pattern = r"\w+"
        tokens = word_tokenize(content)
        cleaned_tokens = self.remove_stopwords(tokens)
        terms = []
        for token in cleaned_tokens:
            if token.isalnum():
                try:
                    term = stemmer.stem(str(token.lower()))
                except:
                    if DEBUG: print("DEBUG | BSBIIndex | token: ", token)
                    term = token.lower()
                terms.append(term)
        return terms

    def parsing_block(self, block_path):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk stemming bahasa Indonesia, seperti
        MpStemmer: https://github.com/ariaghora/mpstemmer 
        Jangan gunakan PySastrawi untuk stemming karena kode yang tidak efisien dan lambat.

        JANGAN LUPA BUANG STOPWORDS! Kalian dapat menggunakan PySastrawi 
        untuk menghapus stopword atau menggunakan sumber lain seperti:
        - Satya (https://github.com/datascienceid/stopwords-bahasa-indonesia)
        - Tala (https://github.com/masdevid/ID-Stopwords)

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_path : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parsing_block(...).
        """
        # TODO

        # get all files in block_path
        files = os.listdir(os.path.join(self.data_dir, block_path))
        td_pairs = []
        # stemmer = MPStemmer()
        for file in files:
            file_path = os.path.join(self.data_dir, block_path, file)
            # get docID
            docID = self.doc_id_map[file_path]
            # get text from file
            with open(file_path, 'r', encoding='utf-8') as f:
                title, text, tags = f.readlines()
                title, text, tags = title.strip(), text.strip(), eval(tags.strip())
                content = title + " " + text
            # get sentences from text
            terms = self.pre_processing_text(content)
            # pattern = r"\w+"
            for term in terms:
                # if token not in stopwords and token.isalnum():
                #     try:
                #         word = stemmer.stem(str(token.lower()))
                #     except:
                #         if DEBUG: print("DEBUG | BSBIIndex | token: ", token)
                #         word = token.lower()
                #     # get termID
                #     termID = self.term_id_map[word]
                #     # append to td_pairs
                #     td_pairs.append((termID, docID))
                # get termID
                termID = self.term_id_map[term]
                # append to td_pairs
                td_pairs.append((termID, docID))

        return td_pairs

    def write_to_index(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-maintain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan strategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        # TODO
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = []
            term_dict[term_id].append(doc_id)
        if DEBUG: print(f"term dict: {dict([(self.term_id_map[term_id], posting_list) for term_id, posting_list in term_dict.items()])}")
        for term_id in sorted(term_dict.keys()):
            doc_tf_counter = Counter()
            doc_tf_counter.update(term_dict[term_id])
            posting_list = sorted(set(term_dict[term_id]))
            # if DEBUG: print(f"term_dict[term_id]: {doc_tf_counter}")
            tf_list = [doc_tf_counter[doc_id] for doc_id in posting_list]
            if DEBUG: print(f"write_to_index() | tf_list: {tf_list} | posting list: {posting_list}")
            index.append(term_id, posting_list, tf_list)
            # index.append(term_id, sorted(list(term_dict[term_id])))

    def merge_index(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi merge_and_sort_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)  # first item
        for t, postings_, tf_list_ in merged_iter:  # from the second item
            if t == curr:
                zip_p_tf = merge_and_sort_posts_and_tfs(list(zip(postings, tf_list)),
                                                        list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k=10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan: 
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        # TODO
        get_score = lambda item: item[1]
        # get all terms in query
        query_terms = self.pre_processing_text(query)

        # get termIDs from query
        self.load()
        # return empty list if query not in term_id_map
        for word in query_terms:
            if word not in self.term_id_map:
                return []
        tfidf_scores = {}
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            N = len(index.doc_length)
            if DEBUG: print(f"N: {N}")
            for term in query_terms:
                # index.reset()
                termId = self.term_id_map[term]
                posting_list = index.get_postings_list(termId)
                tf_list = index.get_tf_list(termId)
                if DEBUG: print(f"term: {term} ({termId}) | posting list: {[self.doc_id_map[doc] for doc in posting_list]} | tf list: {tf_list}")
                df = len(posting_list)
                idf = math.log10(N/df)
                for doc, tf in zip(posting_list, tf_list):
                    score = idf * (1 + math.log10(tf))
                    tfidf_scores[doc] = tfidf_scores.get(doc, 0) + score
                    if DEBUG: print(f"doc : {doc} | tf: {tf} | tfidf_scores : {tfidf_scores}")
        
        sorted_score = sorted(list(tfidf_scores.items()), key=get_score, reverse=True)
        return [(score, self.doc_id_map[doc]) for doc, score in sorted_score[:k]]

    def doc_id_to_path(self, doc_id):
        """
        Mengubah doc_id menjadi nama file yang sesuai

        Parameters
        ----------
        doc_id: int
            doc_id yang ingin diubah

        Returns
        -------
        str
            Nama file yang sesuai dengan doc_id
        """
        self.load()
        return self.doc_id_map[doc_id]


    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan framework TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        """
        # TODO
        get_score = lambda item: item[1]
        # get all terms in query
        query_terms = self.pre_processing_text(query)

        # get termIDs from query
        self.load()
        # return empty list if query not in term_id_map
        for word in query_terms:
            if word not in self.term_id_map:
                return []
        bm25_scores = {}
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            N = len(index.doc_length)
            avdl = index.avdl
            for term in query_terms:
                index.reset()
                termId = self.term_id_map[term]
                posting_list = index.get_postings_list(termId)
                tf_list = index.get_tf_list(termId)
                df = len(posting_list)
                idf = math.log10(N/df)
                for doc, tf in zip(posting_list, tf_list):
                    dl = index.doc_length[doc]
                    score = idf * (k1 + 1) * tf
                    score /= k1 * ((1 - b) + b * dl / avdl) + tf
                    bm25_scores[doc] = bm25_scores.get(doc, 0) + score
        
        sorted_score = sorted(list(bm25_scores.items()), key=get_score, reverse=True)
        return [(score, self.doc_id_map[doc], doc) for doc, score in sorted_score[:k]]

    def retrieve_wand(self, query, k=10, k1=1.2, b=0.75, scoring='bm25'):
        get_score = lambda item: item[1]
        # get all terms in query
        query_terms = self.pre_processing_text(query)

        # get termIDs from query
        self.load()
        # return empty list if query not in term_id_map
        for word in query_terms:
            if word not in self.term_id_map:
                return []
        scoring_method = calculate_bm25 if scoring == 'bm25' else calculate_tfidf
        scores = []
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            N = len(index.doc_length)
            avdl = index.avdl
            term_ids = [self.term_id_map[term] for term in query_terms]
            ubs = [index.get_upper_bound_score(term_id, scoring=scoring) for term_id in term_ids]
            posting_lists = [index.get_postings_list(term_id) for term_id in term_ids]
            tf_lists = [index.get_tf_list(term_id) for term_id in term_ids]
            wand = WAND.from_postings_index(term_ids, ubs, posting_lists, index.doc_length, tf_lists, k, k1, b, avdl, N, scoring_method)
            
            # run the wand algprithm
            while True:
                posting = wand.next()
                if posting == None:
                    break
                
            scores = wand.get_top_k()
        return [(score, self.doc_id_map[doc]) for score, doc in scores]
    def get_doc_sep(self):
        with open(os.path.join(self.output_dir, "doc.sep"), 'rb') as doc_sep:
            return pickle.load(doc_sep)
    def do_indexing(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parsing_block
        untuk parsing dokumen dan memanggil write_to_index yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # Save directory separator
        with open(os.path.join(self.output_dir, "doc.sep"), 'wb') as doc_sep:
            pickle.dump(os.path.sep, doc_sep)

        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parsing_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
                self.write_to_index(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                           for index_id in self.intermediate_indices]
                self.merge_index(indices, merged_index)

    def calculate_deafult_ubs(self):
        """
        Calculate the default upper bound scores for all terms in the index
        """
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as index:
            index.count_all_term_upper_bound_score()
            

def test():

    BSBI_instance = BSBIIndex(data_dir='test_collections',
                              postings_encoding=VBEPostings,
                              output_dir='tmp')
    BSBI_instance.do_indexing() 

    queries = ["tiga kategori dana",
           "10 aktor Skotlandia"]
    for query in queries:
        print("Query  : ", query)
        print("Results:")
        for (score, doc) in BSBI_instance.retrieve_tfidf(query, k=10):
            print(f"{doc:30} {score:>.3f}")
        print()
if __name__ == "__main__":
    DEBUG = sys.argv[1] == 'debug' if len(sys.argv) > 1 else DEBUG

    BSBI_instance = BSBIIndex(data_dir='collections',
                              postings_encoding=VBEPostings,
                              output_dir='index_snowball')
    BSBI_instance.do_indexing()  # memulai indexing!
    # test()