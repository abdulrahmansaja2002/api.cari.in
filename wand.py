import heapq
import math

from compression import VBEPostings
from index import InvertedIndexReader




class TermIterator:
    def __init__(self, postings) -> None:
        self.postings = postings
        self.current_posting = 0

    def start(self):
        return self.postings[0]

    def last(self):
        return self.postings[-1]

    def current(self):
        return self.postings[self.current_posting]

    def next(self):
        if self.current_posting == len(self.postings) - 1:
            return None
        else:
            self.current_posting += 1
            return self.postings[self.current_posting]

    def reset(self):
        self.current_posting = 0
        return self.postings[self.current_posting]

    def next_top(self, pivotid):
        while self.current() != None and self.current().docid < pivotid:
            self.next()
        return self.current()

    def __str__(self) -> str:
        return str([str(posting) for posting in self.postings])


class Posting:
    def __init__(self, docid, score, dl=1, tf=1) -> None:
        self.docid = docid
        self._score = score
        self._dl = dl
        self._tf = tf

    def tf(self):
        return self._tf

    def dl(self):
        return self._dl

    def score(self):
        return self._score

    def __str__(self) -> str:
        return str(self.docid)


class Terms:
    def __init__(self, termId, ub, postings) -> None:
        self.termId = termId
        self.ub = ub
        self.current_posting = 0
        self.postings = postings
        self._df = len(postings)

    def df(self):
        return self._df
    

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_posting == len(self.postings) - 1:
            raise StopIteration
        else:
            self.current_posting += 1
            return self.postings[self.current_posting]

    def start(self):
        self.current_posting = 0
        return self

    def last_posting(self):
        return self.postings[-1]

    def next(self):
        return self.__next__()

    def next_top(self, pivotid):
        while self.current() != None and self.current().docid < pivotid:
            self.next()
        return self.current()

    def current(self):
        return self.postings[self.current_posting]

    def __str__(self) -> str:
        return self.termId
    


def calculate_bm25(postings_iter: Terms, k1=1.2, b=0.75, avdl=1, N=1):
    score = 0
    for iter in postings_iter:
        tf = iter.current().tf()
        dl = iter.current().dl()
        df = iter.df()
        idf = math.log10(N / df)
        score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (dl / avdl))))
    return score


def calculate_tfidf(postings_iter: Terms, k1=1.2, b=0.75, avdl=1, N=1):
    score = 0
    for iter in postings_iter:
        df = iter.df()
        tf = iter.current().tf()
        idf = math.log10(N / df)
        score += idf * (1 + math.log10(tf))
    return score


class WAND:
    default_score = 0

    def __init__(self, terms, k=10, k1=1.2, b=0.75, avdl=1, N=1, scoring=None) -> None:
        self.terms = terms
        self.current_doc = 0
        self.postings = []
        for term in terms:
            term.start()
            self.postings.append(term.__iter__())
        self.k = k
        self.theta = float("-inf")
        self.top_k = []
        self.avdl = avdl
        self.scoring = scoring
        self.N = N
        self.k1 = k1
        self.b = b

    def next(self):
        while True:
            self.sort()
            p_term = self.find_pivot_term()
            if p_term == None:
                return None
            pivotid = p_term.current().docid
            if pivotid == p_term.last_posting().docid:
                return None

            if pivotid <= self.current_doc:
                a_term = self.pick_a_term()
                a_term.next()
            else:
                if self.postings[0].current().docid == pivotid:
                    self.current_doc = pivotid
                    score = (
                        self.scoring(
                            self.postings, self.k1, self.b, self.avdl, self.N
                        )
                        if self.scoring != None
                        else sum(
                            [posting.current().score() for posting in self.postings]
                        )
                    )
                    self.set_theta(score)
                    heapq.heappush(self.top_k, (score, pivotid))
                    return self.current_doc, self.postings
                else:
                    a_term = self.pick_a_term()
                    a_term.next_top(pivotid)

    def sort(self):
        self.postings.sort(key=lambda x: x.current().docid)
        self.terms.sort(key=lambda x: x.current().docid)

    def find_pivot_term(self):
        acum_ub = 0
        for term in self.terms:
            acum_ub += term.ub
            if acum_ub >= self.theta:
                return term

    def pick_a_term(self):
        return self.postings[0]

    def get_top_k(self):
        return heapq.nlargest(self.k, self.top_k)

    def set_theta(self, new_theta):
        if self.theta == float("-inf"):
            self.theta = new_theta
        else:
            self.theta = min(self.theta, new_theta)

    def set_scoring(self, scoring):
        self.scoring = scoring

    def set_avdl(self, avdl):
        self.avdl = avdl

    def set_N(self, N):
        self.N = N

    def set_k1(self, k1):
        self.k1 = k1

    def set_b(self, b):
        self.b = b

    def set_k(self, k):
        self.k = k

    def reset(self):
        self.current_doc = 0
        self.theta = float("-inf")
        self.top_k = []
        for term in self.terms:
            term.iterator.reset()
        for posting in self.postings:
            posting.reset()

    @staticmethod
    def from_postings_index(
        term_ids,
        ubs,
        posting_lists,
        doc_length,
        tf_lists,
        k=10,
        k1=1.2,
        b=0.75,
        avdl=1,
        N=1,
        scoring=calculate_bm25,
    ):
        terms = []
        for term_id, ub, posting_list, tf_list in zip(
            term_ids, ubs, posting_lists, tf_lists
        ):
            postings = [
                Posting(posting, WAND.default_score, tf=tf, dl=doc_length[posting])
                for posting, tf in zip(posting_list, tf_list)
            ]
            terms.append(Terms(term_id, ub, postings))
        return WAND(terms, k, k1, b, avdl, N, scoring)


def test():
    datas = {
        "hujan": [1.6, [(1, 1.5), (2, 0.4), (3, 0.6), (6, 1.0), (8, 1.5), (11, 1.6)]],
        "turun": [1.5, [(1, 0.7), (3, 1.0), (6, 1.5), (8, 1.5), (10, 0.3), (12, 1.1)]],
        "deras": [1.8, [(1, 1.2), (6, 1.0), (7, 0.5), (10, 0.6), (11, 1.8)]],
    }
    terms = {}
    for data in datas:
        ub = datas[data][0]
        postings = [Posting(post[0], post[1]) for post in datas[data][1]]
        terms[data] = Terms(data, ub, postings)
    wand = WAND(list(terms.values()))

    while True:
        posting = wand.next()
        if posting == None:
            break
        # print(posting.docid, posting.score)
    print(wand.get_top_k())


if __name__ == "__main__":
    # test()
    index_name = "main_index"
    posting_encoding = VBEPostings
    directory = "index"
    with InvertedIndexReader(index_name, posting_encoding, directory) as index:
        terms = []
        for term in index.get_terms():
            terms.append(term)
        wand = WAND.from_postings_index(
            [term.term for term in terms],
            [term.ub for term in terms],
            [term.postings for term in terms],
            [term.tfs for term in terms],
            k=10,
            k1=1.2,
            b=0.75,
            avdl=index.get_avdl(),
            N=index.get_N(),
            scoring=calculate_bm25,
        )
        while True:
            posting = wand.next()
            if posting == None:
                break
            # print(posting.docid, posting.score)
        print(wand.get_top_k())
