#===----------------------------------------------------------------------===#
#
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# SOPHON-DEMO is licensed under the 2-Clause BSD License except for the
# third-party components.
#
#===----------------------------------------------------------------------===#
from tqdm import tqdm
from typing import List

from .sentence_model import SentenceModel


class Word2VecEmbedding:
    def __init__(
            self,
            embedding_model_path = "./models/BM1684X/bce_embedding/bce-embedding-base_v1.bmodel",
            tokenizer_path = './models/BM1684X/bce_embedding/token_config',
            max_seq_length = 256,
            device_id = 5
            ):
        self.model = SentenceModel(embedding_model_path=embedding_model_path,
                                   tokenizer_path = tokenizer_path,
                                   max_seq_length=max_seq_length,
                                   device_id=device_id)

    def embed_query(self, text: str) -> List[float]:
        embeddings_tpu = self.model.encode_tpu([text, "", "", ""])
        return embeddings_tpu.tolist()[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings_tpu = self.model.encode_tpu(texts)
        return embeddings_tpu.tolist()

    def docs2embedding(self, docs):
        # now bce-embedding bmodel batch size is 4
        batch_size = 4
        emb = []
        for i in tqdm(range(len(docs) // batch_size)):
            emb += self.embed_documents(docs[i * batch_size: (i + 1) * batch_size])
        if len(docs) % batch_size != 0:
            residue = docs[-(len(docs) % batch_size):] + [" " for _ in range(batch_size - len(docs) % batch_size)]
            emb += self.embed_documents(residue)[:len(docs) % batch_size]
        return emb