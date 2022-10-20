import torch.nn as nn
from transformers import AutoConfig, AutoModel, BertConfig, RobertaModel
import torch
# from coreference.src.metrics import Evaluator, b_cubed, ceafe
from utils import to_cuda, pad_and_stack, to_var
import torch.nn.functional as F
from data import TEMPREL2ID, SUBEVENTREL2ID, CAUSALREL2ID, COREFREL2ID

class EventEncoder(nn.Module):
    def __init__(self, vocab_size, model_name="roberta-base", aggr="mean"):
        nn.Module.__init__(self)
        config = AutoConfig.from_pretrained(model_name)
        if isinstance(config, BertConfig):
            self.model = RobertaModel.from_pretrained(model_name)
        else:
            raise NotImplementedError
        self.model.resize_token_embeddings(vocab_size)
        self.model = nn.DataParallel(self.model)
        # self.embed_linear = nn.Linear(768 * 3, 768)
        self.aggr = aggr
    
    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        event_spans = inputs["event_spans"]
        doc_splits = inputs["splits"]
        event_embed = []
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).last_hidden_state
        # print(output.size())
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
            event_embed.append(torch.stack(embed)) # n_event, embed_dim
        return event_embed

class Score(nn.Module):
    """ Generic scoring module
    """
    def __init__(self, embeds_dim, out_dim=6, hidden_dim=150):
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
    def __init__(self, embed_dim=768, out_dim=6):
        nn.Module.__init__(self)
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.score = Score(embed_dim * 2, out_dim=out_dim)
    
    def forward(self, event_embed):
        # dummy = torch.zeros(self.embed_dim)
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
            # pairs = torch.cat((i_g, j_g, i_g*j_g), dim=1)
            pairs = torch.cat((i_g, j_g), dim=1)
            scores = self.score(pairs)
            # scores = F.softmax(scores, dim=0)
            # scores, _ = pad_and_stack(scores, value=-1000)
            # scores = scores.squeeze(-1)
            all_scores.append(scores)
        all_scores, sizes = pad_and_stack(all_scores, value=-1000) # (doc_num, max_label_num, label_num)
        # print(all_scores.size())
        # print(sizes)
        return all_scores

class CorefPairScorer(nn.Module):
    def __init__(self, embed_dim=768):
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.score = Score(embed_dim * 3, out_dim=1)
    
    def forward(self, event_embed, event_idx):
        # dummy = torch.zeros(self.embed_dim)
        all_probs = []
        for i in range(len(event_embed)):
            embed = event_embed[i]
            embed = embed[event_idx[i]]
            # print(embed.size())
            # filled_labels = to_cuda(self.fill_expand(labels))
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

class Model(nn.Module):
    def __init__(self, vocab_size, model_name="roberta-base", embed_dim=768, aggr="mean"):
        nn.Module.__init__(self)
        self.encoder = EventEncoder(vocab_size, model_name=model_name, aggr=aggr)
        self.temporal_scorer = PairScorer(embed_dim=embed_dim, out_dim=len(TEMPREL2ID))
        self.causal_scorer = PairScorer(embed_dim=embed_dim, out_dim=len(CAUSALREL2ID))
        self.subevent_scorer = PairScorer(embed_dim=embed_dim, out_dim=len(SUBEVENTREL2ID))
        self.coref_scorer = CorefPairScorer(embed_dim=embed_dim)


    def forward(self, inputs):
        output = self.encoder(inputs)
        temporal_output = self.temporal_scorer(output)
        causal_output = self.causal_scorer(output)
        subevent_output = self.subevent_scorer(output)
        coref_output = self.coref_scorer(output, inputs["events_idx"])
        return coref_output, temporal_output, causal_output, subevent_output
        # return output # [[tensor]]