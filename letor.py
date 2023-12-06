import lightgbm as lgb
import numpy as np
import os
from pathlib import Path

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary

from scipy.spatial.distance import cosine

from util import load_data, load_qrels, summarize_text

FILE_PATH = Path(os.path.dirname(os.path.abspath(__file__)))

NUM_NEGATIVES = 1
NUM_LATENT_TOPICS = 200

qrels_folder = FILE_PATH / "qrels"
queries_folder = FILE_PATH / "queries"


MATH_QRELS = qrels_folder / "math.txt"
PHYSICS_QRELS = qrels_folder / "physics.txt"
PRGRAMMERS_QRELS = qrels_folder / "programmers.txt"
GAMING_QRELS = qrels_folder / "gaming.txt"

MATH_QUERIES = queries_folder / "math.txt"
PHYSICS_QUERIES = queries_folder / "physics.txt"
PRGRAMMERS_QUERIES = queries_folder / "programmers.txt"
GAMING_QUERIES = queries_folder / "gaming.txt"



# def vector_rep(text, model, dictionary):
#   rep = [topic_value for (_, topic_value) in model[dictionary.doc2bow(text)]]
#   return rep if len(rep) == NUM_LATENT_TOPICS else [0.] * NUM_LATENT_TOPICS

# def features(query, doc, model, dictionary):
#   v_q = vector_rep(query, model, dictionary)
#   v_d = vector_rep(doc, model, dictionary)
#   q = set(query)
#   d = set(doc)
#   cosine_dist = cosine(v_q, v_d)
#   jaccard = len(q & d) / len(q | d)
#   return v_q + v_d + [jaccard] + [cosine_dist]

# print("Loading Training data...", end="", flush=True)
# qrels = load_qrels(qrel_file=TRAIN_QRELS)
# documents, dataset, group_qid_count = load_data(
#     queries_file=TRAIN_QUERIES,
#     docs_file=TRAIN_DOCS,
#     num_negatives=NUM_NEGATIVES,
#     q_docs_rel=qrels
# )
# print("done!")

# print("Loading Validation data...", end="", flush=True)
# # _, val_dataset, _ = load_data(
# #     queries_file=VALIDATION_QUERIES,
# #     docs_file=TRAIN_DOCS,
# #     num_negatives=NUM_NEGATIVES,
# #     q_docs_rel=load_qrels(qrel_file=VALIDATION_QRELS)
# # )
# print("done!")

# print("Initializing LSI model...", end="", flush=True)
# dictionary = Dictionary()
# bow_corpus = [dictionary.doc2bow(doc, allow_update = True) for doc in documents.values()]
# model = LsiModel(bow_corpus, num_topics = NUM_LATENT_TOPICS) 
# print("done!")


# print("Training...", flush=True)
# X = []
# Y = []
# for (query, doc, rel) in dataset:
#   X.append(features(query, doc, model, dictionary))
#   Y.append(rel)

# # ubah X dan Y ke format numpy array
# X = np.array(X)
# Y = np.array(Y)

# # X_val = []
# # Y_val = []
# # for (query, doc, rel) in val_dataset:
# #   X_val.append(features(query, doc, model, dictionary))
# #   Y_val.append(rel)

# # # ubah X dan Y ke format numpy array
# # X_val, Y_val = np.array(X_val), np.array(Y_val)


# print("Initializing LGBMRanker...", flush=True)
# ranker = lgb.LGBMRanker(
#                     objective="lambdarank",
#                     boosting_type = "gbdt",
#                     n_estimators = 100,
#                     importance_type = "gain",
#                     metric = "ndcg",
#                     num_leaves = 40,
#                     learning_rate = 0.02,
#                     max_depth = -1)
# print("LGBMRanker initialized!")
# print("Fitting...")

# # di contoh kali ini, kita tidak menggunakan validation set
# # jika ada yang ingin menggunakan validation set, silakan saja
# ranker.fit(X, Y,
#            group = group_qid_count,
#           #  eval_set = [(X_val, Y_val)],
#            )
#           #  verbose = 10)

# print("Fitting done!")
# print("Predicting...")

# # test, prediksi terhadap training data itu sendiri
# pred = ranker.predict(X)
# print("Predicting done!")
# print("Predictions: ", pred)

# # save model
# model_name = "model.txt"
# ranker.booster_.save_model(model_name)

class Letor():

  def __init__(self, model_name:str = "model.txt", init_data=True) -> None:
    self.model_name = model_name
    
    if init_data:
      self.init_data()
  def init_data(self):
    print("Loading Training data...", end="", flush=True)

    collections_path = FILE_PATH / "collections"
    math_docs = collections_path / "math"
    physics_docs = collections_path / "physics"
    programmers_docs = collections_path / "programmers"
    gaming_docs = collections_path / "gaming"

    math_documents, math_dataset, math_group_qid_count = load_data(
        queries_file=MATH_QUERIES,
        docs_path=math_docs,
        num_negatives=NUM_NEGATIVES,
        q_docs_rel_file=MATH_QRELS
    )
    physics_documents, physics_dataset, physics_group_qid_count = load_data(
        queries_file=PHYSICS_QUERIES,
        docs_path=physics_docs,
        num_negatives=NUM_NEGATIVES,
        q_docs_rel_file=PHYSICS_QRELS
    )
    programmers_documents, programmers_dataset, programmers_group_qid_count = load_data(
        queries_file=PRGRAMMERS_QUERIES,
        docs_path=programmers_docs,
        num_negatives=NUM_NEGATIVES,
        q_docs_rel_file=PRGRAMMERS_QRELS
    )
    gaming_documents, gaming_dataset, gaming_group_qid_count = load_data(
        queries_file=GAMING_QUERIES,
        docs_path=gaming_docs,
        num_negatives=NUM_NEGATIVES,
        q_docs_rel_file=GAMING_QRELS
    )

    self.data = {
      "math": {
        "documents": math_documents,
        "dataset": math_dataset,
        "group_qid_count": math_group_qid_count
      },
      "physics": {
        "documents": physics_documents,
        "dataset": physics_dataset,
        "group_qid_count": physics_group_qid_count
      },
      "programmers": {
        "documents": programmers_documents,
        "dataset": programmers_dataset,
        "group_qid_count": programmers_group_qid_count
      },
      "gaming": {
        "documents": gaming_documents,
        "dataset": gaming_dataset,
        "group_qid_count": gaming_group_qid_count
      }
    }

    print("done!")

    print("Loading Validation data...", end="", flush=True)
    # _, val_dataset, _ = load_data(
    #     queries_file=VALIDATION_QUERIES,
    #     docs_file=TRAIN_DOCS,
    #     num_negatives=NUM_NEGATIVES,
    #     q_docs_rel=load_qrels(qrel_file=VALIDATION_QRELS)
    # )
    print("done!")

    print("Initializing LSI model...", end="", flush=True)
    self.dictionary = Dictionary()
    documents = {**math_documents, **physics_documents, **programmers_documents, **gaming_documents}
    bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in documents.values()]
    self.model = LsiModel(bow_corpus, num_topics = NUM_LATENT_TOPICS) 
    print("done!")

  def vector_rep(self, text):
    rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
    return rep if len(rep) == NUM_LATENT_TOPICS else [0.] * NUM_LATENT_TOPICS

  def features(self, query, doc):
    v_q = self.vector_rep(query)
    v_d = self.vector_rep(doc)
    q = set(query)
    d = set(doc)
    cosine_dist = cosine(v_q, v_d)
    jaccard = len(q & d) / len(q | d)

    # summarize features
    summarize_doc = summarize_text(" ".join(doc))
    v_summarize = self.vector_rep(summarize_doc.split())
    cosine_summarize = cosine(v_q, v_summarize)
    jaccard_summarize = len(q & set(summarize_doc)) / len(q | set(summarize_doc))


    return [jaccard] + [cosine_dist] + [jaccard_summarize] + [cosine_summarize]
  def train(self):
    print("Initializing LGBMRanker...", flush=True)
    ranker = lgb.LGBMRanker(
                        objective="lambdarank",
                        boosting_type = "gbdt",
                        n_estimators = 100,
                        importance_type = "gain",
                        metric = "ndcg",
                        num_leaves = 40,
                        learning_rate = 0.02,
                        max_depth = -1)
    print("LGBMRanker initialized!")
    print("Fitting...")


    print("Training Math dataset...", flush=True)
    X = []
    Y = []
    for (query, doc, rel) in self.data["math"]["dataset"]:
      X.append(self.features(query, doc))
      Y.append(rel)

    # ubah X dan Y ke format numpy array
    X = np.array(X)
    Y = np.array(Y)

    # fit
    ranker.fit(X, Y,
              group = self.data["math"]["group_qid_count"],
              #  eval_set = [(X_val, Y_val)],
              )
              #  verbose = 10)

    print("Training Physics dataset...", flush=True)
    X = []
    Y = []
    for (query, doc, rel) in self.data["physics"]["dataset"]:
      X.append(self.features(query, doc))
      Y.append(rel)

    # ubah X dan Y ke format numpy array
    X = np.array(X)
    Y = np.array(Y)

    # fit
    ranker.fit(X, Y,
              group = self.data["physics"]["group_qid_count"],
              #  eval_set = [(X_val, Y_val)],
              )
              #  verbose = 10)

    print("Training Programmers dataset...", flush=True)
    X = []
    Y = []

    for (query, doc, rel) in self.data["programmers"]["dataset"]:
      X.append(self.features(query, doc))
      Y.append(rel)

    # ubah X dan Y ke format numpy array
    X = np.array(X)
    Y = np.array(Y)

    # fit
    ranker.fit(X, Y,
              group = self.data["programmers"]["group_qid_count"],
              #  eval_set = [(X_val, Y_val)],
              )
              #  verbose = 10)

    print("Training Gaming dataset...", flush=True)
    X = []
    Y = []
    for (query, doc, rel) in self.data["gaming"]["dataset"]:
      X.append(self.features(query, doc))
      Y.append(rel)

    # ubah X dan Y ke format numpy array
    X = np.array(X)
    Y = np.array(Y)

    # fit
    ranker.fit(X, Y,
              group = self.data["gaming"]["group_qid_count"],
              #  eval_set = [(X_val, Y_val)],
              )
              #  verbose = 10)


    # X_val = []
    # Y_val = []
    # for (query, doc, rel) in val_dataset:
    #   X_val.append(features(query, doc, model, dictionary))
    #   Y_val.append(rel)

    # # ubah X dan Y ke format numpy array
    # X_val, Y_val = np.array(X_val), np.array(Y_val)



    # di contoh kali ini, kita tidak menggunakan validation set
    # jika ada yang ingin menggunakan validation set, silakan saja

    self.ranker = ranker
    return ranker
  
  def save_model(self, model_file=None):
    if self.ranker == None:
      print("Ranker is not ready. please do train or load model (with as_ranker param True) first!")
      return None
    if model_file == None:
      model_file = FILE_PATH / 'model' / self.model_name
    
    dict_file = FILE_PATH / 'model' / 'dictionary'
    lsi_file = FILE_PATH / 'model' / 'lsi_model'
    self.ranker.booster_.save_model(model_file)
    self.dictionary.save(str(dict_file))
    self.model.save(str(lsi_file))
  
  def load_model(self, model_file=None, as_ranker=False):
    if model_file == None:
      model_file = FILE_PATH / 'model' / self.model_name

    dict_file = FILE_PATH / 'model' / 'dictionary'
    lsi_file = FILE_PATH / 'model' / 'lsi_model'
    ranker = lgb.Booster(model_file=model_file)
    self.dictionary = Dictionary.load(str(dict_file))
    self.model = LsiModel.load(str(lsi_file))
    if as_ranker:
      self.ranker = ranker
    return ranker
  
  def re_ranking(self, query:str, list_doc:list[str]):
    if self.ranker == None:
      print("Ranker is not ready. please do train or load model (with as_ranker param True) first!")
      return None
    X = []
    splited_query = query.split()
    dids = []
    for doc in list_doc:
      did, *content = doc.split()
      dids.append(did)
      X.append(self.features(splited_query, content))
    
    X = np.array(X)

    scores = self.ranker.predict(X)
    return sorted([(did, score) for (did, score) in zip(dids, scores)], key=lambda pair: pair[1], reverse=True)

def test():
  letor = Letor(init_data=False)
  # letor.train()
  # letor.save_model()
  letor.load_model(as_ranker=True)
  print(letor.re_ranking("what is the meaning of life", ["1 this is a test", "2 this is another test", "3 this is the last test"]))


if __name__ == "__main__":
  letor = Letor()
  letor.train()
  letor.save_model()  