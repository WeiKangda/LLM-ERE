import torch
import torch.nn as nn
import torch.nn.functional as F
from .gat_layers import GraphAttentionLayer, SpGraphAttentionLayer
from transformers import AutoConfig, AutoModel, BertConfig, RobertaModel
from .utils import to_cuda, pad_and_stack

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x
        #x = F.elu(self.out_att(x, adj))
        #return F.log_softmax(x, dim=1)


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, adj)
        return x
        #x = F.elu(self.out_att(x, adj))
        #return F.log_softmax(x, dim=1)

class EventEncoder(nn.Module):
    def __init__(self, vocab_size, model_name="/scratch/user/kangda/MAVEN-ERE/roberta-base", aggr="mean"):
        nn.Module.__init__(self)
        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, BertConfig):
            self.model = RobertaModel.from_pretrained(model_name)
            #for param in self.model.encoder.parameters():
            #    param.requires_grad = False
        else:
            raise NotImplementedError
        self.model.resize_token_embeddings(vocab_size)
        self.model = nn.DataParallel(self.model)
        self.aggr = aggr
    
    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        event_spans = inputs["event_spans"]
        doc_splits = inputs["splits"]
        event_embed = []
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).last_hidden_state
        for i in range(0, len(doc_splits)-1):
            embed = []
            doc_embed = output[doc_splits[i]:doc_splits[i+1]]
            doc_spans = event_spans[i]
            for j, spans in enumerate(doc_spans):
                for span in spans:
                    if self.aggr == "max":
                        embed.append(doc_embed[j][span[0]:span[1]].max(0)[0])
                    elif self.aggr == "mean":
                        embed.append(doc_embed[j][span[0]:span[1]].mean(0))
                    else:
                        raise NotImplementedError
            event_embed.append(torch.stack(embed))
        return event_embed

class Score(nn.Module):
    """ Generic scoring module
    """
    def __init__(self, embeds_dim, out_dim=7, hidden_dim=150):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embeds_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        """ Output a scalar score for an input x """
        return self.score(x)

class PairScorer(nn.Module):
    def __init__(self, embed_dim=768, out_dim=7):
        nn.Module.__init__(self)
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.gat = GAT(embed_dim, nhid=embed_dim, nclass=embed_dim, dropout=0.6, alpha=0.2, nheads=1)
        self.score = Score(embed_dim * 2, out_dim=out_dim)

    def forward(self, event_embed):
        all_scores = []
        for i in range(len(event_embed)):
            embed = event_embed[i]
            if embed.size(0) <= 1: # no event or only one event
                scores = to_cuda(torch.tensor([[-1000] * self.out_dim]))
                all_scores.append(scores)
                continue
            
            #print(len(embed))
            #print(embed.shape)
            cos = nn.CosineSimilarity(dim=0, eps=1e-6)
            adj = to_cuda(torch.zeros(len(embed), len(embed)))
            for i in range(len(embed)):
                for j in range(len(embed)):
                    #if ((i < int(len(embed) * 0.2) and j < int(len(embed) * 0.2)) \
                    #    or (i < int(len(embed) * 0.2) and j > len(embed) - int(len(embed) * 0.2)) \
                    #    or (i > len(embed) - int(len(embed) * 0.2) and j < int(len(embed) * 0.2)) \
                    #    or (i > len(embed) - int(len(embed) * 0.2) and j > len(embed) - int(len(embed) * 0.2))) and i != j:
                    #    adj[i][j] = 1
                    if cos(embed[i], embed[j]) >= 0.5 and i != j:
                        adj[i][j] == 1
            #print(adj)
            output = self.gat(embed, adj)
            #print(output.shape)

            event1_id, event2_id = zip(*[(index, k) for index in range(len(embed)) for k in range(len(embed)) if index != k])
            event1_id, event2_id = to_cuda(torch.tensor(event1_id)), to_cuda(torch.tensor(event2_id))
            #print(event1_id.shape)
            #print(event2_id.shape)
            i_g = torch.index_select(output, 0, event1_id)
            j_g = torch.index_select(output, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)
            #print(i_g.shape)
            #print(j_g.shape)
            #print(pairs.shape)
            scores = self.score(pairs)
            all_scores.append(scores)
        all_scores, sizes = pad_and_stack(all_scores, value=-1000)
        return all_scores

class Model_GAT(nn.Module):
    def __init__(self, vocab_size, out_dim=7, model_name="/scratch/user/kangda/MAVEN-ERE/roberta-base", embed_dim=768, aggr="mean"):
        nn.Module.__init__(self)
        self.encoder = EventEncoder(vocab_size, model_name=model_name, aggr=aggr)
        self.scorer = PairScorer(embed_dim=embed_dim, out_dim=out_dim)

    def forward(self, inputs):
        output = self.encoder(inputs)
        output = self.scorer(output)
        return output