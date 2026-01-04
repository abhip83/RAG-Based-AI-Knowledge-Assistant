from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def context_precision(retrieved_texts, ground_truth, embedding_model):
    retrieved_embeddings = embedding_model.embed_documents(retrieved_texts)
    gt_embedding = embedding_model.embed_query(ground_truth)

    similarities = cosine_similarity(
        retrieved_embeddings,
        [gt_embedding]
    )

    return float(np.mean(similarities))


def faithfulness(answer, context, embedding_model):
    ans_emb = embedding_model.embed_query(answer)
    ctx_emb = embedding_model.embed_query(context)

    score = cosine_similarity([ans_emb], [ctx_emb])[0][0]
    return float(score)
