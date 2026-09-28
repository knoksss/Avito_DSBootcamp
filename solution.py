import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

TOPK = 50
W_MC, W_LOC, W_POP = 1.0, 0.05, 0.02
KNN = 30
USE_DENSE = False
CHUNK = 256

train = pd.read_parquet("train.parquet")
bq = pd.read_parquet("benchmark_queries.parquet")
bi = pd.read_parquet("benchmark_items.parquet")


def s(col):
    return col.fillna("").astype(str)


def item_text(df):
    # заголовок повторяем 2 раза: он важнее описания; описание обрезаем
    return (s(df.item_title_raw) + " " + s(df.item_title_raw) + " " +
            s(df.item_infm_params_text) + " " + s(df.item_description_raw).str[:400]
            ).str.lower()


def query_text(df):
    # текст запроса + текстовые фильтры
    return (s(df.search_query) + " " + s(df.search_infm_params_text)).str.lower()


tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                        sublinear_tf=True, dtype=np.float32, max_features=1_500_000)
X_items = tfidf.fit_transform(item_text(bi))
Q_bench = tfidf.transform(query_text(bq))


mcs = pd.Index(pd.concat([bi.item_microcat_id, train.item_microcat_id]).unique())
mc_train = mcs.get_indexer(train.item_microcat_id)
mc_items = mcs.get_indexer(bi.item_microcat_id)

# уникальные train-запросы и распределение выбранных ими подкатегорий
tq = pd.DataFrame({"q": query_text(train), "mc": mc_train})
uq = tq.groupby("q").ngroup().values
n_uq = uq.max() + 1
M = sp.csr_matrix((np.ones(len(tq), np.float32), (uq, tq.mc.values)),
                  shape=(n_uq, len(mcs)))
M = normalize(M, norm="l1")
uq_text = tq.drop_duplicates("q").sort_values("q")["q"]
uq_text = tq.groupby("q").size().index
T_train = tfidf.transform(uq_text)


def mc_prior(Qv):
    sim = (Qv @ T_train.T).toarray()
    idx = np.argpartition(-sim, KNN, axis=1)[:, :KNN]
    w = np.take_along_axis(sim, idx, 1) ** 3
    rows = np.repeat(np.arange(len(sim)), KNN)
    W = sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=sim.shape)
    P = (W @ M).toarray()
    return P / (P.sum(1, keepdims=True) + 1e-9)


pop = (bi.item_rating.fillna(0).values * np.log1p(bi.item_rating_reviews_count.fillna(0).values))
pop = (pop / (pop.max() + 1e-9)).astype(np.float32)
item_loc = bi.item_location_id.values


if USE_DENSE:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("intfloat/multilingual-e5-base")   # локальная open-source модель
    E_items = st.encode(["passage: " + t for t in item_text(bi)], batch_size=128,
                        normalize_embeddings=True, show_progress_bar=True)
    E_q = st.encode(["query: " + t for t in query_text(bq)], batch_size=128,
                    normalize_embeddings=True)


def retrieve(qdf, Qv, E_q=None):
    ids = bi.item_id.values
    loc_q = qdf.search_location_id.values
    out = []
    for a in range(0, Qv.shape[0], CHUNK):
        b = min(a + CHUNK, Qv.shape[0])
        score = (Qv[a:b] @ X_items.T).toarray()
        if USE_DENSE and E_q is not None:
            score = 0.5 * score + 0.5 * (E_q[a:b] @ E_items.T)
        score += W_MC * mc_prior(Qv[a:b])[:, mc_items]
        score += W_LOC * (item_loc[None, :] == loc_q[a:b, None])
        score += W_POP * pop[None, :]
        top = np.argpartition(-score, TOPK, axis=1)[:, :TOPK]
        out += [list(ids[t]) for t in top]
    return out


preds = retrieve(bq, Q_bench, E_q if USE_DENSE else None)


answer = pd.DataFrame({
    "query_id": bq.query_id.values,
    "answer": [" ".join(map(str, p)) for p in preds],
})
answer.to_csv("answer.csv", index=False)


assert answer.query_id.is_unique and len(answer) == len(bq)
assert all(len(p) == len(set(p)) <= TOPK for p in preds)
assert set(np.concatenate(preds)) <= set(bi.item_id)
print("ok", answer.shape)