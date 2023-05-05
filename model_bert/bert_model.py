import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import constant
from utils import torch_utils

from transformers import BertModel, BertTokenizer, BertConfig
# model_config = BertConfig.from_pretrained("./huggingface/bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("./huggingface/bert-base-uncased")
model = BertModel.from_pretrained("./huggingface/bert-base-uncased").to(torch.device('cuda:0'))



class bert_UtterEncoder(nn.Module):
    """
    Encoder for each utterance by bert
    """

    def __init__(self, config, vocab):
        super().__init__()
        self.config = config
        self.vocab = vocab
        # Embedding
        self.embedding = nn.Embedding(self.vocab.n_words, self.config.embed_dim)
        self.init_pretrained_embeddings_from_numpy(np.load(open(config.embed_f, 'rb'), allow_pickle=True))
        self.embeddings = self.embedding

    def init_pretrained_embeddings_from_numpy(self, pretrained_word_vectors):
        self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        if self.config.tune_topk <= 0:
            print("Do not fine tune word embedding layer")
            self.embedding.weight.requires_grad = False
        elif self.config.tune_topk < self.vocab.n_words:
            print(f"Finetune top {self.config.tune_topk} word embeddings")
            self.embedding.weight.register_hook(lambda x: torch_utils.keep_partial_grad(x, self.config.tune_topk))
        else:
            print("Finetune all word embeddings")

    def forward(self, batch):
        """
        :conv_bert: the token of bert_tokenizer.
        """

        conv_bert = batch['conv_bert']
        batch, num_utt, num_words = batch['conv_batch'].shape
        utter_rep = torch.zeros([batch, num_utt, constant.BERT_DIM]).to(device='cuda')

        for i, tokens in enumerate(conv_bert):
            bert_rep = model(tokens)
            length, _ = tokens.shape
            for j in range(length):
                utter_rep[i][j] = bert_rep.last_hidden_state[j][0]

        return utter_rep
