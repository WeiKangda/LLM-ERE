import torch.nn as nn
from transformers import AutoConfig, AutoModel, BertConfig, RobertaModel
from torch_geometric.nn.conv import RGCNConv
import torch
from .utils import to_cuda, pad_and_stack, to_var, get_predicted_clusters, get_event2cluster, fill_expand
import torch.nn.functional as F
from .data import TEMPREL2ID, SUBEVENTREL2ID, CAUSALREL2ID, COREFREL2ID
import os
import yaml

class RGCN(torch.nn.Module):
    def __init__(self, num_relations, dropout):
        super(RGCN, self).__init__()

        self.num_relations = num_relations
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 768))
        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(768, 768, num_relations, num_bases=None)
        self.conv2 = RGCNConv(768, 768, num_relations, num_bases=None)

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type):
        #x = self.entity_embedding(entity)
        x = F.relu(self.conv1(entity, edge_index, edge_type))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = self.conv2(x, edge_index, edge_type)
        #print(f"RGCN forward output shape: {x.shape}")
        return x

    def distmult(self, embedding, triplets):
        s = torch.index_select(embedding, 0, triplets[:,0])
        r = torch.index_select(self.relation_embedding, 0, triplets[:,1])
        o = torch.index_select(embedding, 0, triplets[:,2])
        score = torch.sum(s * r * o, dim=1)
        score = score.view(-1, self.num_relations)
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
            for param in self.model.base_model.parameters():
                param.requires_grad = False
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
    def __init__(self, embeds_dim, out_dim=6, hidden_dim=200):
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

class CorefPairScorer(nn.Module):
    def __init__(self, embed_dim=768):
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.score = Score(embed_dim * 3, out_dim=1)
    
    def forward(self, event_embed, event_idx):
        all_probs = []
        for i in range(len(event_embed)):
            embed = event_embed[i]
            embed = embed[event_idx[i]]
            if embed.size(0) > 1: # at least one event
                event_id, antecedent_id = zip(*[(index, k) for index in range(len(embed)) for k in range(index)])
                event_id, antecedent_id = to_cuda(torch.tensor(event_id)), to_cuda(torch.tensor(antecedent_id))
                i_g = torch.index_select(embed, 0, event_id)
                j_g = torch.index_select(embed, 0, antecedent_id)
                pairs = torch.cat((i_g, j_g, i_g*j_g), dim=1)
                s_ij = self.score(pairs)
                split_scores = [to_cuda(torch.tensor([]))] \
                            + list(torch.split(s_ij, [i for i in range(len(embed)) if i], dim=0)) # first event has no valid antecedent
            else:
                split_scores = [to_cuda(torch.tensor([]))]
            epsilon = to_var(torch.tensor([[0.]])) # dummy score default to 0.0
            with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores] # dummy index default to same index as itself 
            probs = [F.softmax(tensr, dim=0) for tensr in with_epsilon]
            probs, _ = pad_and_stack(probs, value=-1000)
            probs = probs.squeeze(-1)
            all_probs.append(probs)
        return all_probs

class Model_RGCN(nn.Module):
    def __init__(self, vocab_size, model_name="/scratch/user/kangda/MAVEN-ERE/roberta-base", embed_dim=768, aggr="mean", out_dim=10):
        nn.Module.__init__(self)
        self.out_dim = out_dim
        # Create encoder layer for each event category and load the best baseline encoder for each category
        self.temporal_encoder = EventEncoder(vocab_size, model_name=model_name, aggr=aggr)
        self.causal_encoder = EventEncoder(vocab_size, model_name=model_name, aggr=aggr)
        self.subevent_encoder = EventEncoder(vocab_size, model_name=model_name, aggr=aggr)
        self.coref_encoder = EventEncoder(vocab_size, model_name=model_name, aggr=aggr)

        state = torch.load(os.path.join("./output/baseline/42/MAVEN-ERE", "best_TEMPORAL_roberta"))
        self.temporal_encoder.model.load_state_dict(state["model"])
        state = torch.load(os.path.join("./output/baseline/42/MAVEN-ERE", "best_CAUSAL_roberta"))
        self.causal_encoder.model.load_state_dict(state["model"])
        state = torch.load(os.path.join("./output/baseline/42/MAVEN-ERE", "best_SUBEVENT_roberta"))
        self.subevent_encoder.model.load_state_dict(state["model"])
        state = torch.load(os.path.join("./output/baseline/42/MAVEN-ERE", "best_COREFERENCE_roberta"))
        self.coref_encoder.model.load_state_dict(state["model"])

        self.rgcn = RGCN(10, 0.4) # number of relation is calculated by adding the number of relationships of each category and exclude none type

        # Create classification layer for each event category and load the best baseline classification layer for each category
        self.temporal_scorer = Score(embed_dim * 2, out_dim=len(TEMPREL2ID))
        self.causal_scorer = Score(embed_dim * 2, out_dim=len(CAUSALREL2ID))
        self.subevent_scorer = Score(embed_dim * 2, out_dim=len(SUBEVENTREL2ID))
        self.coref_scorer= Score(embed_dim * 3, out_dim=1)

        state = torch.load(os.path.join("./output/baseline/42/MAVEN-ERE", "best_TEMPORAL_linear"))
        self.temporal_scorer.load_state_dict(state["model"])
        state = torch.load(os.path.join("./output/baseline/42/MAVEN-ERE", "best_CAUSAL_linear"))
        self.causal_scorer.load_state_dict(state["model"])
        state = torch.load(os.path.join("./output/baseline/42/MAVEN-ERE", "best_SUBEVENT_linear"))
        self.subevent_scorer.load_state_dict(state["model"])
        #state = torch.load(os.path.join("./output/baseline/42/MAVEN-ERE", "best_COREFERENCE_linear"))
        #self.coref_scorer.score.load_state_dict(state["model"])

    # train on gold graph
    def forward(self, inputs):
        temporal_labels = inputs["temporal_labels"].view(-1)
        causal_labels = inputs["causal_labels"].view(-1)
        subevent_labels = inputs["subevent_labels"].view(-1)
        coref_labels = inputs["coref_labels"]
        
        temporal_output = self.temporal_encoder(inputs)
        causal_output = self.causal_encoder(inputs)
        subevent_output = self.subevent_encoder(inputs)
        coref_output = self.coref_encoder(inputs)        
        
        event_idx = inputs["events_idx"]
        all_probs = []
        for i in range(len(coref_output)):
            embed = coref_output[i]
            embed = embed[event_idx[i]]
            if embed.size(0) > 1: # at least one event
                event_id, antecedent_id = zip(*[(index, k) for index in range(len(embed)) for k in range(index)])
                event_id, antecedent_id = to_cuda(torch.tensor(event_id)), to_cuda(torch.tensor(antecedent_id))
                i_g = torch.index_select(embed, 0, event_id)
                j_g = torch.index_select(embed, 0, antecedent_id)
                pairs = torch.cat((i_g, j_g, i_g*j_g), dim=1)
                s_ij = self.coref_scorer(pairs)
                split_scores = [to_cuda(torch.tensor([]))] \
                            + list(torch.split(s_ij, [i for i in range(len(embed)) if i], dim=0)) # first event has no valid antecedent
            else:
                split_scores = [to_cuda(torch.tensor([]))]
            epsilon = to_var(torch.tensor([[0.]])) # dummy score default to 0.0
            with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores] # dummy index default to same index as itself 
            probs = [F.softmax(tensr, dim=0) for tensr in with_epsilon]
            probs, _ = pad_and_stack(probs, value=-1000)
            probs = probs.squeeze(-1)
            
            pred_clusters, pred_event2cluster = get_predicted_clusters(probs)

            edge_index = torch.empty((2, 0)).to(torch.int64)
            edge_type = torch.empty(0).to(torch.int64)
            for cluster in pred_clusters:
                if len(cluster) <= 1:
                    continue
                for event_1 in cluster:
                    for event_2 in cluster:
                        edge_index = torch.cat((edge_index, torch.tensor([[event_1], [event_2]]).to(torch.int64)), dim=1) 
                        edge_type = torch.cat((edge_type, torch.tensor([9]).to(torch.int64)), dim=0) 
            edge_index = to_cuda(edge_index)
            edge_type = to_cuda(edge_type)
            #print(f"edge_index: {edge_index}")
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"edge_type: {edge_type}")
            #print(f"edge_type shape: {edge_type.shape}")
            embed = self.rgcn(embed, edge_index, edge_type)
            if embed.size(0) > 1: # at least one event
                event_id, antecedent_id = zip(*[(index, k) for index in range(len(embed)) for k in range(index)])
                event_id, antecedent_id = to_cuda(torch.tensor(event_id)), to_cuda(torch.tensor(antecedent_id))
                i_g = torch.index_select(embed, 0, event_id)
                j_g = torch.index_select(embed, 0, antecedent_id)
                pairs = torch.cat((i_g, j_g, i_g*j_g), dim=1)
                s_ij = self.coref_scorer(pairs)
                split_scores = [to_cuda(torch.tensor([]))] \
                            + list(torch.split(s_ij, [i for i in range(len(embed)) if i], dim=0)) # first event has no valid antecedent
            else:
                split_scores = [to_cuda(torch.tensor([]))]
            epsilon = to_var(torch.tensor([[0.]])) # dummy score default to 0.0
            with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores] # dummy index default to same index as itself 
            probs = [F.softmax(tensr, dim=0) for tensr in with_epsilon]
            probs, _ = pad_and_stack(probs, value=-1000)
            probs = probs.squeeze(-1)
            all_probs.append(probs)

        temporal_all_scores = []
        for i in range(len(temporal_output)):
            embed = temporal_output[i]
            if embed.size(0) <= 1: # no event or only one event
                scores = to_cuda(torch.tensor([[-1000] * self.out_dim]))
                temporal_all_scores.append(scores)
                continue
            event1_id, event2_id = zip(*[(index, k) for index in range(len(embed)) for k in range(len(embed)) if index != k])
            event1_id, event2_id = to_cuda(torch.tensor(event1_id)), to_cuda(torch.tensor(event2_id))
            i_g = torch.index_select(embed, 0, event1_id)
            j_g = torch.index_select(embed, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.temporal_scorer(pairs)
            scores = scores.view(-1, scores.size(-1))
            noise_pred = torch.argmax(scores, dim=-1)

            event1_id, event2_id = event1_id.unsqueeze(0), event2_id.unsqueeze(0)
            edge_index = torch.cat((event1_id, event2_id), dim=0)
            #print(f"embed shape: {embed.shape}")
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"temporal_labels shape: {temporal_labels.shape}")
            edge_index = edge_index[:, noise_pred != 6]
            edge_type = noise_pred[noise_pred != 6]
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"edge_type shape: {edge_type.shape}")
            output = self.rgcn(embed, edge_index, edge_type)
            event1_id, event2_id = event1_id.squeeze(0), event2_id.squeeze(0)
            i_g = torch.index_select(output, 0, event1_id)
            j_g = torch.index_select(output, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.temporal_scorer(pairs)
            temporal_all_scores.append(scores)

        temporal_all_scores, sizes = pad_and_stack(temporal_all_scores, value=-1000)

        causal_all_scores = []
        for i in range(len(causal_output)):
            embed = causal_output[i]
            if embed.size(0) <= 1: # no event or only one event
                scores = to_cuda(torch.tensor([[-1000] * self.out_dim]))
                causal_all_scores.append(scores)
                continue
            event1_id, event2_id = zip(*[(index, k) for index in range(len(embed)) for k in range(len(embed)) if index != k])
            event1_id, event2_id = to_cuda(torch.tensor(event1_id)), to_cuda(torch.tensor(event2_id))
            i_g = torch.index_select(embed, 0, event1_id)
            j_g = torch.index_select(embed, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.causal_scorer(pairs)
            scores = scores.view(-1, scores.size(-1))
            noise_pred = torch.argmax(scores, dim=-1)

            event1_id, event2_id = event1_id.unsqueeze(0), event2_id.unsqueeze(0)
            edge_index = torch.cat((event1_id, event2_id), dim=0)
            #print(f"embed shape: {embed.shape}")
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"causal_labels shape: {causal_labels.shape}")
            edge_index = edge_index[:, noise_pred != 0]
            edge_type = noise_pred[noise_pred != 0]
            edge_type[edge_type == 1] = 6
            edge_type[edge_type == 2] = 7
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"edge_type shape: {edge_type.shape}")
            output = self.rgcn(embed, edge_index, edge_type)
            event1_id, event2_id = event1_id.squeeze(0), event2_id.squeeze(0)
            i_g = torch.index_select(output, 0, event1_id)
            j_g = torch.index_select(output, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.causal_scorer(pairs)
            causal_all_scores.append(scores)

        causal_all_scores, sizes = pad_and_stack(causal_all_scores, value=-1000)

        subevent_all_scores = []
        for i in range(len(subevent_output)):
            embed = subevent_output[i]
            if embed.size(0) <= 1: # no event or only one event
                scores = to_cuda(torch.tensor([[-1000] * self.out_dim]))
                subevent_all_scores.append(scores)
                continue
            event1_id, event2_id = zip(*[(index, k) for index in range(len(embed)) for k in range(len(embed)) if index != k])
            event1_id, event2_id = to_cuda(torch.tensor(event1_id)), to_cuda(torch.tensor(event2_id))
            i_g = torch.index_select(embed, 0, event1_id)
            j_g = torch.index_select(embed, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.subevent_scorer(pairs)
            scores = scores.view(-1, scores.size(-1))
            noise_pred = torch.argmax(scores, dim=-1)

            event1_id, event2_id = event1_id.unsqueeze(0), event2_id.unsqueeze(0)
            edge_index = torch.cat((event1_id, event2_id), dim=0)
            #print(f"embed shape: {embed.shape}")
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"subevent_labels shape: {subevent_labels.shape}")
            edge_index = edge_index[:, noise_pred != 0]
            edge_type = noise_pred[noise_pred != 0]
            edge_type[edge_type == 1] = 8
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"edge_type shape: {edge_type.shape}")
            output = self.rgcn(embed, edge_index, edge_type)
            event1_id, event2_id = event1_id.squeeze(0), event2_id.squeeze(0)
            i_g = torch.index_select(output, 0, event1_id)
            j_g = torch.index_select(output, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.subevent_scorer(pairs)
            subevent_all_scores.append(scores)

        subevent_all_scores, sizes = pad_and_stack(subevent_all_scores, value=-1000)

        #coref_output = self.coref_scorer(coref_output, inputs["events_idx"])
        coref_output = all_probs
        temporal_output = temporal_all_scores
        causal_output = causal_all_scores
        subevent_output = subevent_all_scores
        return coref_output, temporal_output, causal_output, subevent_output

    # infer with noisy graph
    def predict(self, inputs):
        temporal_output = self.temporal_encoder(inputs)
        causal_output = self.causal_encoder(inputs)
        subevent_output = self.subevent_encoder(inputs)
        coref_output = self.coref_encoder(inputs)        
        
        event_idx = inputs["events_idx"]
        all_probs = []
        for i in range(len(coref_output)):
            embed = coref_output[i]
            embed = embed[event_idx[i]]
            if embed.size(0) > 1: # at least one event
                event_id, antecedent_id = zip(*[(index, k) for index in range(len(embed)) for k in range(index)])
                event_id, antecedent_id = to_cuda(torch.tensor(event_id)), to_cuda(torch.tensor(antecedent_id))
                i_g = torch.index_select(embed, 0, event_id)
                j_g = torch.index_select(embed, 0, antecedent_id)
                pairs = torch.cat((i_g, j_g, i_g*j_g), dim=1)
                s_ij = self.coref_scorer(pairs)
                split_scores = [to_cuda(torch.tensor([]))] \
                            + list(torch.split(s_ij, [i for i in range(len(embed)) if i], dim=0)) # first event has no valid antecedent
            else:
                split_scores = [to_cuda(torch.tensor([]))]
            epsilon = to_var(torch.tensor([[0.]])) # dummy score default to 0.0
            with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores] # dummy index default to same index as itself 
            probs = [F.softmax(tensr, dim=0) for tensr in with_epsilon]
            probs, _ = pad_and_stack(probs, value=-1000)
            probs = probs.squeeze(-1)
            
            pred_clusters, pred_event2cluster = get_predicted_clusters(probs)

            edge_index = torch.empty((2, 0)).to(torch.int64)
            edge_type = torch.empty(0).to(torch.int64)
            for cluster in pred_clusters:
                if len(cluster) <= 1:
                    continue
                for event_1 in cluster:
                    for event_2 in cluster:
                        edge_index = torch.cat((edge_index, torch.tensor([[event_1], [event_2]]).to(torch.int64)), dim=1) 
                        edge_type = torch.cat((edge_type, torch.tensor([9]).to(torch.int64)), dim=0) 
            edge_index = to_cuda(edge_index)
            edge_type = to_cuda(edge_type)
            #print(f"edge_index: {edge_index}")
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"edge_type: {edge_type}")
            #print(f"edge_type shape: {edge_type.shape}")
            embed = self.rgcn(embed, edge_index, edge_type)
            if embed.size(0) > 1: # at least one event
                event_id, antecedent_id = zip(*[(index, k) for index in range(len(embed)) for k in range(index)])
                event_id, antecedent_id = to_cuda(torch.tensor(event_id)), to_cuda(torch.tensor(antecedent_id))
                i_g = torch.index_select(embed, 0, event_id)
                j_g = torch.index_select(embed, 0, antecedent_id)
                pairs = torch.cat((i_g, j_g, i_g*j_g), dim=1)
                s_ij = self.coref_scorer(pairs)
                split_scores = [to_cuda(torch.tensor([]))] \
                            + list(torch.split(s_ij, [i for i in range(len(embed)) if i], dim=0)) # first event has no valid antecedent
            else:
                split_scores = [to_cuda(torch.tensor([]))]
            epsilon = to_var(torch.tensor([[0.]])) # dummy score default to 0.0
            with_epsilon = [torch.cat((score, epsilon), dim=0) for score in split_scores] # dummy index default to same index as itself 
            probs = [F.softmax(tensr, dim=0) for tensr in with_epsilon]
            probs, _ = pad_and_stack(probs, value=-1000)
            probs = probs.squeeze(-1)
            all_probs.append(probs)

        temporal_all_scores = []
        for i in range(len(temporal_output)):
            embed = temporal_output[i]
            if embed.size(0) <= 1: # no event or only one event
                scores = to_cuda(torch.tensor([[-1000] * self.out_dim]))
                temporal_all_scores.append(scores)
                continue
            event1_id, event2_id = zip(*[(index, k) for index in range(len(embed)) for k in range(len(embed)) if index != k])
            event1_id, event2_id = to_cuda(torch.tensor(event1_id)), to_cuda(torch.tensor(event2_id))
            i_g = torch.index_select(embed, 0, event1_id)
            j_g = torch.index_select(embed, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.temporal_scorer(pairs)
            scores = scores.view(-1, scores.size(-1))
            noise_pred = torch.argmax(scores, dim=-1)

            event1_id, event2_id = event1_id.unsqueeze(0), event2_id.unsqueeze(0)
            edge_index = torch.cat((event1_id, event2_id), dim=0)
            #print(f"embed shape: {embed.shape}")
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"temporal_labels shape: {temporal_labels.shape}")
            #print(f"noise_pred shape: {noise_pred.shape}")
            #print(f"noise_pred: {noise_pred}")
            edge_index = edge_index[:, noise_pred != 6]
            edge_type = noise_pred[noise_pred != 6]
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"edge_type shape: {edge_type.shape}")
            #print(f"edge_type: {edge_type}")
            output = self.rgcn(embed, edge_index, edge_type)
            event1_id, event2_id = event1_id.squeeze(0), event2_id.squeeze(0)
            i_g = torch.index_select(output, 0, event1_id)
            j_g = torch.index_select(output, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.temporal_scorer(pairs)
            temporal_all_scores.append(scores)

        temporal_all_scores, sizes = pad_and_stack(temporal_all_scores, value=-1000)

        causal_all_scores = []
        for i in range(len(causal_output)):
            embed = causal_output[i]
            if embed.size(0) <= 1: # no event or only one event
                scores = to_cuda(torch.tensor([[-1000] * self.out_dim]))
                causal_all_scores.append(scores)
                continue
            event1_id, event2_id = zip(*[(index, k) for index in range(len(embed)) for k in range(len(embed)) if index != k])
            event1_id, event2_id = to_cuda(torch.tensor(event1_id)), to_cuda(torch.tensor(event2_id))
            i_g = torch.index_select(embed, 0, event1_id)
            j_g = torch.index_select(embed, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.causal_scorer(pairs)
            scores = scores.view(-1, scores.size(-1))
            noise_pred = torch.argmax(scores, dim=-1)

            event1_id, event2_id = event1_id.unsqueeze(0), event2_id.unsqueeze(0)
            edge_index = torch.cat((event1_id, event2_id), dim=0)
            #print(f"embed shape: {embed.shape}")
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"noise_pred shape: {noise_pred.shape}")
            #print(f"noise_pred: {noise_pred}")
            edge_index = edge_index[:, noise_pred != 0]
            edge_type = noise_pred[noise_pred != 0]
            edge_type[edge_type == 1] = 6
            edge_type[edge_type == 2] = 7
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"edge_type shape: {edge_type.shape}")
            #print(f"edge_type: {edge_type}")
            output = self.rgcn(embed, edge_index, edge_type)
            event1_id, event2_id = event1_id.squeeze(0), event2_id.squeeze(0)
            i_g = torch.index_select(output, 0, event1_id)
            j_g = torch.index_select(output, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.causal_scorer(pairs)
            causal_all_scores.append(scores)

        causal_all_scores, sizes = pad_and_stack(causal_all_scores, value=-1000)

        subevent_all_scores = []
        for i in range(len(subevent_output)):
            embed = subevent_output[i]
            if embed.size(0) <= 1: # no event or only one event
                scores = to_cuda(torch.tensor([[-1000] * self.out_dim]))
                subevent_all_scores.append(scores)
                continue
            event1_id, event2_id = zip(*[(index, k) for index in range(len(embed)) for k in range(len(embed)) if index != k])
            event1_id, event2_id = to_cuda(torch.tensor(event1_id)), to_cuda(torch.tensor(event2_id))
            i_g = torch.index_select(embed, 0, event1_id)
            j_g = torch.index_select(embed, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.subevent_scorer(pairs)
            scores = scores.view(-1, scores.size(-1))
            noise_pred = torch.argmax(scores, dim=-1)

            event1_id, event2_id = event1_id.unsqueeze(0), event2_id.unsqueeze(0)
            edge_index = torch.cat((event1_id, event2_id), dim=0)
            #print(f"embed shape: {embed.shape}")
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"noise_pred shape: {noise_pred.shape}")
            #print(f"noise_pred: {noise_pred}")
            edge_index = edge_index[:, noise_pred != 0]
            edge_type = noise_pred[noise_pred != 0]
            edge_type[edge_type == 1] = 8
            #print(f"edge_index shape: {edge_index.shape}")
            #print(f"edge_type shape: {edge_type.shape}")
            #print(f"edge_type: {edge_type}")
            output = self.rgcn(embed, edge_index, edge_type)
            event1_id, event2_id = event1_id.squeeze(0), event2_id.squeeze(0)
            i_g = torch.index_select(output, 0, event1_id)
            j_g = torch.index_select(output, 0, event2_id)
            pairs = torch.cat((i_g, j_g), dim=1)

            scores = self.subevent_scorer(pairs)
            subevent_all_scores.append(scores)

        subevent_all_scores, sizes = pad_and_stack(subevent_all_scores, value=-1000)

        
        #coref_output = self.coref_scorer(coref_output, inputs["events_idx"])
        coref_output = all_probs
        temporal_output = temporal_all_scores
        causal_output = causal_all_scores
        subevent_output = subevent_all_scores
        return coref_output, temporal_output, causal_output, subevent_output