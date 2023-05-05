import numpy as np
import torch
import torch
import torch.nn as nn
import dgl


from model_lstm.gat import WSWGAT
from utils import constant

class ConvEncoder(nn.Module):
    """
    without sent2sent and add residual connection
    adapted from brxx122/hetersumgraph/higraph.py
    """
    def __init__(self, config, embeddings):
        super().__init__()

        self.config = config
        # self._embed = embed
        # self.embed_size = hps.word_emb_dim
        # self.word_embedding = embeddings
        self.word_embedding, _, _ = embeddings
        self.ner_embedding = nn.Embedding(len(constant.ner_i2s), self.config.ner_embed_dim)
        self.speaker_embedding = nn.Embedding(10, self.config.embed_dim) # assume there is only 10 speaker in the conversation
        self.arg_embedding = nn.Embedding(2, self.config.embed_dim)
        self.dep_embedding = nn.Embedding(len(constant.dep_i2s),self.config.dep_embed_dim)


        # sent node feature
        self.ws_embed = nn.Embedding(len(constant.pos_i2s), config.edge_embed_size) # bucket = 10
        self.wn_embed = nn.Embedding(config.wn_edge_bucket, config.edge_embed_size) # bucket = 10
        self.wp_embed = nn.Embedding(len(constant.dep_i2s), config.edge_embed_size)
        self.sent_feature_proj = nn.Linear(config.glstm_hidden_dim*2, config.ggcn_hidden_size, bias=False)
        self.ner_feature_proj = nn.Linear(config.ner_embed_dim, config.ggcn_hidden_size, bias=False)
        self.dep_feature_proj = nn.Linear(config.dep_embed_dim, config.ggcn_hidden_size, bias=False)

        self.glstm = nn.LSTM(self.config.gcn_lin_dim,
                                self.config.glstm_hidden_dim,
                                num_layers=self.config.glstm_layers, dropout=0.1,
                                batch_first=True, bidirectional=True)


        # word -> sent
        self.word2sent = WSWGAT(in_dim=config.embed_dim,
                                out_dim=config.ggcn_hidden_size,
                                num_heads=config.word2sent_n_head,
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="W2S"
                                )

        # sent -> word
        self.sent2word = WSWGAT(in_dim=config.ggcn_hidden_size,
                                out_dim=config.embed_dim,
                                num_heads=config.sent2word_n_head, # make sure word_embedding divisible by this number
                                attn_drop_out=config.atten_dropout_prob,
                                ffn_inner_hidden_size=config.ffn_inner_hidden_size,
                                ffn_drop_out=config.ffn_dropout_prob,
                                feat_embed_size=config.edge_embed_size, # for edge
                                layerType="S2W"
                                )

        # node classification
        self.wh = nn.Linear(config.ggcn_hidden_size, 2)

    def forward(self, graph, batch, local_feature):
        supernode_id = graph.filter_nodes(lambda nodes: nodes.data['unit'] == 1) # supernodes contains nerNode and sentNode
        #parsenode_id = graph.filter_nodes(lambda nodes: nodes.data['unit'] == 2)
        # Initialize states
        self.set_wordNode_feature(graph)
        self.set_speakerNode_feature(graph)
        self.set_argNode_feature(graph)
        self.set_wordSentEdge_feature(graph)
        # [snode, glstm_hidden_dim] -> [snode, n_hidden_size]
        self.set_sentNode_feature_lstm(graph, batch, local_feature)
        self.set_wordNerEdge_feature(graph)
        self.set_wordDepEdge_feature(graph)  # word-dep edge
        self.set_nerNode_feature(graph)
        self.set_depNode_feature(graph)

        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["unit"]==0) # both word node and speaker node
        # the start state
        word_state = graph.nodes[wnode_id].data['embed']
        sent_state = graph.nodes[supernode_id].data['init_state']
        ner_state = graph.nodes[supernode_id].data['ner_init_state']
        # dep_state
        dep_state = graph.nodes[supernode_id].data['dep_init_state']

        # GAT
        word_state = self.sent2word(graph, word_state, sent_state)
        if self.config.dep_or_ner == 'ner':
            ner_state = self.word2sent(graph, word_state, ner_state)
            word_state = self.sent2word(graph, word_state, ner_state)
        elif self.config.dep_or_ner == 'dep':
            dep_state = self.word2sent(graph, word_state, dep_state)
            word_state = self.sent2word(graph, word_state, dep_state)
        else:  # all
            ner_state = self.word2sent(graph, word_state, ner_state)
            dep_state = self.word2sent(graph, word_state, dep_state)
            # dep_state = self.word2sent(graph, word_state, dep_state)
            word_state = self.sent2word(graph, word_state, ner_state)
            word_state = self.sent2word(graph, word_state, dep_state)
            # word_state = self.sent2word(graph, word_state, dep_state)


        for i in range(self.config.ggcn_n_iter):
            sent_state = self.word2sent(graph, word_state, sent_state)
            word_state = self.sent2word(graph, word_state, sent_state)
        graph.nodes[wnode_id].data["feat"] = word_state
        return None

    def set_wordNode_feature(self, graph):
        wnode_id = graph.filter_nodes(lambda nodes: nodes.data["dtype"]==0) # only word node
        wid = graph.nodes[wnode_id].data["id"]  # [n_wnodes]
        w_embed = self.word_embedding(wid)  # [n_wnodes, D]
        graph.nodes[wnode_id].data["embed"] = w_embed
        return w_embed

    def set_speakerNode_feature(self, graph):
        speakernode_id = graph.filter_nodes(lambda nodes: nodes.data['dtype']==3) # only speaker node
        speakerid = graph.nodes[speakernode_id].data['id']
        speaker_embed = self.speaker_embedding(speakerid)
        graph.nodes[speakernode_id].data["embed"] = speaker_embed

    def set_argNode_feature(self, graph):
        argnode_id = graph.filter_nodes(lambda nodes: nodes.data['dtype']==4) # only arg node
        argid = graph.nodes[argnode_id].data['id']
        arg_embed = self.arg_embedding(argid)
        graph.nodes[argnode_id].data['embed'] = arg_embed
    
    def set_wordSentEdge_feature(self, graph):
        # Intialize word sent edge
        wsedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 0)   # both word node and speaker node
        ws_edge = graph.edges[wsedge_id].data['ws_link']
        graph.edges[wsedge_id].data["ws_embed"] = self.ws_embed(ws_edge)


    def set_sentNode_feature_lstm(self, graph, batch, local_feature):
        sent_feature, _ = self.glstm(local_feature)  # (batch, max_number_utt, glstm_hidden_dim*2)
        sent_feature = sent_feature * batch['conv_mask'][:, :, 0].unsqueeze(-1)  # masking
        sent_feature = sent_feature.reshape(-1, sent_feature.size(-1))
        sent_feature = sent_feature[batch['utter_index']]  # (batch * total_number_utt, glstm_hidden_dim*2)

        snode_id = graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 1)  # only sent node

        graph.nodes[snode_id].data['init_state'] = self.sent_feature_proj(sent_feature)


    def set_nerNode_feature(self, graph):
        nnode_id = graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 2) # only ner node
        nerid = graph.nodes[nnode_id].data['id'] # [n_nerNodes]
        ner_embed = self.ner_embedding(nerid)
        graph.nodes[nnode_id].data['ner_init_state'] = self.ner_feature_proj(ner_embed)

    def set_depNode_feature(self, graph):
        depnode_id = graph.filter_nodes(lambda nodes: nodes.data['dtype'] == 5) # only ner node
        depid = graph.nodes[depnode_id].data['id'] # [n_nerNodes]
        dep_embed = self.dep_embedding(depid)
        graph.nodes[depnode_id].data['dep_init_state'] = self.dep_feature_proj(dep_embed)

    def set_wordNerEdge_feature(self, graph):
        wnedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 1) # only word node and NER node
        wn_edge = graph.edges[wnedge_id].data['wn_link']
        graph.edges[wnedge_id].data['wn_embed'] = self.wn_embed(wn_edge)

    def set_wordDepEdge_feature(self, graph):
        
        wpedge_id = graph.filter_edges(lambda edges: edges.data["dtype"] == 1) # only word node and Dep node
        wp_edge = graph.edges[wpedge_id].data['wp_link']
        graph.edges[wp_edge].data['wp_embed'] = self.wp_embed(wp_edge)


