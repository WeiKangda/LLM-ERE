import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import RGCNConv
from transformers import AutoConfig, AutoModel, BertConfig, RobertaModel
from .utils import to_cuda, pad_and_stack, uniform
from .model import Model


class RGCN(torch.nn.Module):
    def __init__(self, num_relations, dropout):
        super(RGCN, self).__init__()

        self.num_relations = num_relations
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 768))
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(768, 768, num_relations, num_bases=None)
        self.conv2 = RGCNConv(768, 768, num_relations, num_bases=None)
        #self.conv3 = RGCNConv(768, 768, num_relations, num_bases=None)
        #self.conv4 = RGCNConv(768, 768, num_relations, num_bases=None)

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type):
        #x = self.entity_embedding(entity)
        x = F.relu(self.conv1(entity, edge_index, edge_type))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = self.conv2(x, edge_index, edge_type)
        #x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        #x = F.relu(self.conv3(x, edge_index, edge_type))
        #x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        #x = self.conv4(x, edge_index, edge_type)
        # = self.conv1(entity, edge_index, edge_type)
        #print(f"RGCN forward output shape: {x.shape}")
        return x

    def distmult(self, embedding, triplets):
        #print(triplets[:,0])
        #print(triplets[:,1])
        #print(triplets[:,2])
        s = torch.index_select(embedding, 0, triplets[:,0])
        r = torch.index_select(self.relation_embedding, 0, triplets[:,1])
        o = torch.index_select(embedding, 0, triplets[:,2])
        score = torch.sum(s * r * o, dim=1)
        score = score.view(-1, self.num_relations)
        #pred = torch.argmax(score, dim=1)
        #print(f"distmult output shape: {score.shape}")
        #print(f"pred output shape: {pred.shape}")
        
        return score

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)
        y = to_cuda(torch.zeros(score.shape[0], score.shape[1]))
        y[range(y.shape[0]), target]=1
        return F.binary_cross_entropy_with_logits(score, y), score

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))

    def loss(self, embedding, triplets, target, reg_ratio=1e-2):
        score_loss, score = self.score_loss(embedding, triplets, target)
        reg_loss = reg_ratio*self.reg_loss(embedding)

        return score_loss + reg_loss, score

class EventEncoder(nn.Module):
    def __init__(self, vocab_size, model_name="/scratch/user/kangda/MAVEN-ERE/roberta-base", aggr="mean"):
        nn.Module.__init__(self)
        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, BertConfig):
            self.model = RobertaModel.from_pretrained(model_name)
            state = torch.load(os.path.join("./output/baseline/0/maven_ignore_none_True_None", "best_roberta"))
            #print(state)
            self.model.load_state_dict(state["model"])
            #for param in self.model.base_model.parameters():
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
        self.score = RGCN(out_dim, 0.4)
        self.baseline = Score(embed_dim * 2, out_dim=out_dim)
        state = torch.load(os.path.join("./output/baseline/0/maven_ignore_none_True_None", "linear"))
        self.baseline.load_state_dict(state["model"])
        for param in self.baseline.parameters():
                param.requires_grad = False
        self.classifier = Score(embed_dim * 2, out_dim=out_dim)
        state = torch.load(os.path.join("./output/baseline/0/maven_ignore_none_True_None", "linear"))
        self.classifier.load_state_dict(state["model"])

    def forward(self, event_embed, labels):
        all_scores = []
        for i in range(len(event_embed)):
            embed = event_embed[i]
            if embed.size(0) <= 1: # no event or only one event
                scores = to_cuda(torch.tensor([[-1000] * self.out_dim]))
                all_scores.append(scores)
                continue
            event1_id, event2_id = zip(*[(index, k) for index in range(len(embed)) for k in range(len(embed)) if index != k])
            event1_id, event2_id = to_cuda(torch.tensor(event1_id)), to_cuda(torch.tensor(event2_id))
            i_g = torch.index_select(embed, 0, event1_id)
            j_g = torch.index_select(embed, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.baseline(pairs)
            #print(f"scores shape: {scores.shape}")
            scores = scores.view(-1, scores.size(-1))
            #print(f"scores shape: {scores.shape}")
            noise_pred = torch.argmax(scores, dim=-1)
            #print(f"noise_pred shape: {noise_pred.shape}")

            #r = torch.arange(self.out_dim)
            #triplets = torch.empty((0, 3)).to(torch.int32)
            #for i in range(len(event1_id)):
            #    for j in range(len(r)):
            #        triplets = torch.cat((triplets, torch.tensor([event1_id[i], r[j], event2_id[i]]).unsqueeze(0).to(torch.int32)), dim=0) 
            #triplets = to_cuda(triplets)
            event1_id, event2_id = event1_id.unsqueeze(0), event2_id.unsqueeze(0)
            edge_index = torch.cat((event1_id, event2_id), dim=0)
            #print(f"edge_index shape: {edge_index.shape}")
            #edge_index = edge_index[:, noise_pred != 6]
            #edge_type = noise_pred[noise_pred != 6]
            edge_index = edge_index[:, labels != 6]
            edge_type = labels[labels != 6]
            #edge_type = to_cuda(torch.tensor(labels))
            #print(f"embed shape: {embed.shape}")
            #print(f"edge_type shape: {edge_type.shape}")
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"triplets shape: {triplets.shape}")
            #print(f"labels shape: {labels.shape}")
            output = self.score(embed, edge_index, edge_type)
            #scores, preds = self.score.distmult(output, triplets)
            #loss, scores = self.score.loss(output, triplets, labels)
            event1_id, event2_id = event1_id.squeeze(0), event2_id.squeeze(0)
            i_g = torch.index_select(output, 0, event1_id)
            j_g = torch.index_select(output, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.classifier(pairs)
            all_scores.append(scores)

        all_scores, sizes = pad_and_stack(all_scores, value=-1000)
        return all_scores#, loss

class Model_RGCN(nn.Module):
    def __init__(self, vocab_size, out_dim=7, model_name="/scratch/user/kangda/MAVEN-ERE/roberta-base", embed_dim=768, aggr="mean"):
        nn.Module.__init__(self)
        self.out_dim = out_dim
        self.encoder = EventEncoder(vocab_size, model_name=model_name, aggr=aggr)
        self.scorer = PairScorer(embed_dim=embed_dim, out_dim=out_dim)

    def forward(self, inputs, labels):
        output = self.encoder(inputs)
        output = self.scorer(output, labels)
        return output

    def predict(self, inputs):
        encoder_output = self.encoder(inputs)
        all_scores = []
        for i in range(len(encoder_output)):
            embed = encoder_output[i]
            if embed.size(0) <= 1: # no event or only one event
                scores = to_cuda(torch.tensor([[-1000] * self.out_dim]))
                all_scores.append(scores)
                continue
            event1_id, event2_id = zip(*[(index, k) for index in range(len(embed)) for k in range(len(embed)) if index != k])
            event1_id, event2_id = to_cuda(torch.tensor(event1_id)), to_cuda(torch.tensor(event2_id))
            i_g = torch.index_select(embed, 0, event1_id)
            j_g = torch.index_select(embed, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.scorer.baseline(pairs)
            
            #print(f"scores shape: {scores.shape}")
            scores = scores.view(-1, scores.size(-1))
            #print(f"scores shape: {scores.shape}")
            noise_pred = torch.argmax(scores, dim=-1)
            #print(f"noise_pred shape: {noise_pred.shape}")

            #r = torch.arange(self.out_dim)
            #triplets = torch.empty((0, 3)).to(torch.int32)
            #for i in range(len(event1_id)):
            #    for j in range(len(r)):
            #        triplets = torch.cat((triplets, torch.tensor([event1_id[i], r[j], event2_id[i]]).unsqueeze(0).to(torch.int32)), dim=0) 
            #triplets = to_cuda(triplets)
            event1_id, event2_id = event1_id.unsqueeze(0), event2_id.unsqueeze(0)
            edge_index = torch.cat((event1_id, event2_id), dim=0)
            edge_index = edge_index[:, noise_pred != 6]
            edge_type = noise_pred[noise_pred != 6]
            output = self.scorer.score(embed, edge_index, edge_type)
            #scores = self.scorer.score.distmult(output, triplets)
            event1_id, event2_id = event1_id.squeeze(0), event2_id.squeeze(0)
            i_g = torch.index_select(output, 0, event1_id)
            j_g = torch.index_select(output, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.scorer.classifier(pairs)
            all_scores.append(scores)

        all_scores, sizes = pad_and_stack(all_scores, value=-1000)
        return all_scores