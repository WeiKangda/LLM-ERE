#from src.rgcn import *
from src.rgcn2 import *
from src.model import *
from src.utils import to_cuda, to_var
import torch
import random
import numpy as np
from tqdm import tqdm
from src.data2 import myDataset, get_dataloader
from transformers import AdamW, RobertaTokenizer, get_linear_schedule_with_warmup
from src.utils import get_predicted_clusters, get_event2cluster, fill_expand
from src.metrics import evaluate_documents, b_cubed, ceafe, muc, Evaluator, blanc
from src.dump_result import coref_dump,causal_dump,temporal_dump,subevent_dump
import argparse
from torch.optim import Adam
import torch.nn as nn
from sklearn.metrics import classification_report
from src.data2 import TEMPREL2ID, ID2TEMPREL, CAUSALREL2ID, ID2CAUSALREL, SUBEVENTREL2ID, ID2SUBEVENTREL
import warnings
import os
import sys
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

class EvalResult:
    def __init__(self, gold, mention_to_gold, clusters, mention_to_cluster):
        self.gold = gold
        self.mention_to_gold = mention_to_gold
        self.clusters = clusters
        self.mention_to_cluster = mention_to_cluster


def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def draw_gold_graph(data):
    G = nx.Graph()
        
    labeldict = {}
    for mention in data['mentionid2mention'][0]:
        if mention.startswith("TIME"):
            G.add_node(mention, sent_id=data['mentionid2mention'][0][mention]['sent_id'], word=data['mentionid2mention'][0][mention]['mention'])
            labeldict[mention] = (data['mentionid2mention'][0][mention]['mention'], data['mentionid2mention'][0][mention]['sent_id'])
        else:
            G.add_node(mention, sent_id=data['mentionid2mention'][0][mention]['sent_id'], word=data['mentionid2mention'][0][mention]['trigger_word'])
            labeldict[mention] = (data['mentionid2mention'][0][mention]['trigger_word'], data['mentionid2mention'][0][mention]['sent_id'])
        
    data["temporal_labels"] = data["temporal_labels"].cpu().numpy().tolist()
    temporal_edge_labels = {}
    for i, pair in enumerate(data['event_id_pairs'][0]):
        if data["temporal_labels"][0][i] != 6 and data["temporal_labels"][0][i] != -100:
            G.add_edge(pair[0], pair[1])
            temporal_edge_labels[pair] = ID2TEMPREL[data["temporal_labels"][0][i]]
        
    data["causal_labels"] = data["causal_labels"].cpu().numpy().tolist()
    causal_edge_labels = {}
    for i, pair in enumerate(data['event_id_pairs'][0]):
        if data["causal_labels"][0][i] != 0 and data["causal_labels"][0][i] != -100:
            G.add_edge(pair[0], pair[1])
            causal_edge_labels[pair] = ID2CAUSALREL[data["causal_labels"][0][i]]

    data["subevent_labels"] = data["subevent_labels"].cpu().numpy().tolist()
    subevent_edge_labels = {}
    for i, pair in enumerate(data['event_id_pairs'][0]):
        if data["subevent_labels"][0][i] != 0 and data["subevent_labels"][0][i] != -100:
            G.add_edge(pair[0], pair[1])
            subevent_edge_labels[pair] = ID2SUBEVENTREL[data["subevent_labels"][0][i]]

    plt.figure(figsize=(15,15))
    pos = nx.spring_layout(G, seed=0, scale=4, k=7/np.sqrt(len(G.nodes())), iterations=20)
    nx.draw_networkx(G, pos, labels=labeldict, with_labels = True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = temporal_edge_labels, font_size=6)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = causal_edge_labels, font_size=6)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = subevent_edge_labels, font_size=6)

    plt.savefig(f"./images/gold_graph_{data['doc_id'][0]}.png")
    plt.close()

def draw_predicted_graph(data, temporal_pred, causal_pred, subevent_pred):
    G = nx.Graph()
        
    labeldict = {}
    for mention in data['mentionid2mention'][0]:
        if mention.startswith("TIME"):
            G.add_node(mention, sent_id=data['mentionid2mention'][0][mention]['sent_id'], word=data['mentionid2mention'][0][mention]['mention'])
            labeldict[mention] = (data['mentionid2mention'][0][mention]['mention'], data['mentionid2mention'][0][mention]['sent_id'])
        else:
            G.add_node(mention, sent_id=data['mentionid2mention'][0][mention]['sent_id'], word=data['mentionid2mention'][0][mention]['trigger_word'])
            labeldict[mention] = (data['mentionid2mention'][0][mention]['trigger_word'], data['mentionid2mention'][0][mention]['sent_id'])
        
    temporal_pred = temporal_pred.cpu().numpy().tolist()
    temporal_edge_labels = {}
    for i, pair in enumerate(data['event_id_pairs'][0]):
        if temporal_pred[i] != 6 and temporal_pred[i] >= 0:
            G.add_edge(pair[0], pair[1])
            temporal_edge_labels[pair] = ID2TEMPREL[temporal_pred[i]]
        
    causal_pred = causal_pred.cpu().numpy().tolist()
    causal_edge_labels = {}
    for i, pair in enumerate(data['event_id_pairs'][0]):
        if causal_pred[i] != 0 and causal_pred[i] >= 0:
            G.add_edge(pair[0], pair[1])
            causal_edge_labels[pair] = ID2CAUSALREL[causal_pred[i]]

    subevent_pred = subevent_pred.cpu().numpy().tolist()
    subevent_edge_labels = {}
    for i, pair in enumerate(data['event_id_pairs'][0]):
        if subevent_pred[i] != 0 and subevent_pred[i] >= 0:
            G.add_edge(pair[0], pair[1])
            subevent_edge_labels[pair] = ID2SUBEVENTREL[subevent_pred[i]]

    plt.figure(figsize=(15,15))
    pos = nx.spring_layout(G, seed=0, scale=4, k=7/np.sqrt(len(G.nodes())), iterations=20)
    nx.draw_networkx(G, pos, labels=labeldict, with_labels = True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = temporal_edge_labels, font_size=6)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = causal_edge_labels, font_size=6)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = subevent_edge_labels, font_size=6)

    plt.savefig(f"./images/predicted_graph_{data['doc_id'][0]}.png")
    plt.close()

def draw_comparison_graph(data, temporal_pred, causal_pred, subevent_pred):
    G = nx.Graph()
        
    labeldict = {}
    for mention in data['mentionid2mention'][0]:
        if mention.startswith("TIME"):
            G.add_node(mention, sent_id=data['mentionid2mention'][0][mention]['sent_id'], word=data['mentionid2mention'][0][mention]['mention'])
            labeldict[mention] = (data['mentionid2mention'][0][mention]['mention'], data['mentionid2mention'][0][mention]['sent_id'])
        else:
            G.add_node(mention, sent_id=data['mentionid2mention'][0][mention]['sent_id'], word=data['mentionid2mention'][0][mention]['trigger_word'])
            labeldict[mention] = (data['mentionid2mention'][0][mention]['trigger_word'], data['mentionid2mention'][0][mention]['sent_id'])
        
    temporal_edge_labels = {}
    temporal_correct_edges = []
    temporal_wrong_edges = []
    temporal_wrong_color = []
    temporal_correct_color = []
    temporal_pred = temporal_pred.cpu().numpy().tolist()
    assert len(data["temporal_labels"][0]) == len(temporal_pred)
    assert len(temporal_pred) == len(data['event_id_pairs'][0])
    for i, pair in enumerate(data['event_id_pairs'][0]):
        if data["temporal_labels"][0][i] != -100:
            if data["temporal_labels"][0][i] != temporal_pred[i] and data["temporal_labels"][0][i] == 6:
                G.add_edge(pair[0], pair[1])
                temporal_edge_labels[pair] = ID2TEMPREL[temporal_pred[i]]
                temporal_wrong_edges.append(pair)
                temporal_wrong_color.append('yellow')
            elif data["temporal_labels"][0][i] != temporal_pred[i] and data["temporal_labels"][0][i] != 6:
                G.add_edge(pair[0], pair[1])
                temporal_edge_labels[pair] = (ID2TEMPREL[data["temporal_labels"][0][i]], ID2TEMPREL[temporal_pred[i]])
                temporal_wrong_edges.append(pair)
                temporal_wrong_color.append('red')
            elif data["temporal_labels"][0][i] == temporal_pred[i] and data["temporal_labels"][0][i] != 6:
                G.add_edge(pair[0], pair[1])
                temporal_edge_labels[pair] = ID2TEMPREL[data["temporal_labels"][0][i]]
                temporal_correct_edges.append(pair)
                temporal_correct_color.append('black')
        
    causal_edge_labels = {}
    causal_correct_edges = []
    causal_wrong_edges = []
    causal_wrong_color = []
    causal_correct_color = []
    causal_pred = causal_pred.cpu().numpy().tolist()
    assert len(data["causal_labels"][0]) == len(causal_pred)
    assert len(causal_pred) == len(data['event_id_pairs'][0])
    for i, pair in enumerate(data['event_id_pairs'][0]):
        if data["causal_labels"][0][i] != -100:
            if data["causal_labels"][0][i] != causal_pred[i] and data["causal_labels"][0][i] == 0:
                G.add_edge(pair[0], pair[1])
                causal_edge_labels[pair] = ID2CAUSALREL[causal_pred[i]]
                causal_wrong_edges.append(pair)
                causal_wrong_color.append('yellow')
            elif data["causal_labels"][0][i] != causal_pred[i] and data["causal_labels"][0][i] != 0:
                G.add_edge(pair[0], pair[1])
                causal_edge_labels[pair] = (ID2CAUSALREL[data["causal_labels"][0][i]], ID2CAUSALREL[causal_pred[i]])
                causal_wrong_edges.append(pair)
                causal_wrong_color.append('red')
            elif data["causal_labels"][0][i] == causal_pred[i] and data["causal_labels"][0][i] != 0:
                G.add_edge(pair[0], pair[1])
                causal_edge_labels[pair] = ID2CAUSALREL[data["causal_labels"][0][i]]
                causal_correct_edges.append(pair)
                causal_correct_color.append('black')
    
    subevent_edge_labels = {}
    subevent_correct_edges = []
    subevent_wrong_edges = []
    subevent_wrong_color = []
    subevent_correct_color = []
    subevent_pred = subevent_pred.cpu().numpy().tolist()
    assert len(data["subevent_labels"][0]) == len(subevent_pred)
    assert len(subevent_pred) == len(data['event_id_pairs'][0])
    for i, pair in enumerate(data['event_id_pairs'][0]):
        if data["subevent_labels"][0][i] != -100:
            if data["subevent_labels"][0][i] != subevent_pred[i] and data["subevent_labels"][0][i] == 0:
                G.add_edge(pair[0], pair[1])
                subevent_edge_labels[pair] = ID2SUBEVENTREL[subevent_pred[i]]
                subevent_wrong_edges.append(pair)
                subevent_wrong_color.append('yellow')
            elif data["subevent_labels"][0][i] != subevent_pred[i] and data["subevent_labels"][0][i] != 0:
                G.add_edge(pair[0], pair[1])
                subevent_edge_labels[pair] = (ID2SUBEVENTREL[data["subevent_labels"][0][i]], ID2SUBEVENTREL[subevent_pred[i]])
                subevent_wrong_edges.append(pair)
                subevent_wrong_color.append('red')
            elif data["subevent_labels"][0][i] == subevent_pred[i] and data["subevent_labels"][0][i] != 0:
                G.add_edge(pair[0], pair[1])
                subevent_edge_labels[pair] = ID2SUBEVENTREL[data["subevent_labels"][0][i]]
                subevent_correct_edges.append(pair)
                subevent_correct_color.append('black')

    plt.figure(figsize=(15,15))
    pos = nx.spring_layout(G, seed=0, scale=4, k=7/np.sqrt(len(G.nodes())), iterations=20)
    nx.draw_networkx(G, pos, labels=labeldict, with_labels = True)
    nx.draw_networkx_edges(G, pos, edgelist=temporal_correct_edges, edge_color=temporal_correct_color)
    nx.draw_networkx_edges(G, pos, edgelist=temporal_wrong_edges, edge_color=temporal_wrong_color)
    nx.draw_networkx_edges(G, pos, edgelist=causal_correct_edges, edge_color=causal_correct_color)
    nx.draw_networkx_edges(G, pos, edgelist=causal_wrong_edges, edge_color=causal_wrong_color)
    nx.draw_networkx_edges(G, pos, edgelist=subevent_correct_edges, edge_color=subevent_correct_color)
    nx.draw_networkx_edges(G, pos, edgelist=subevent_wrong_edges, edge_color=subevent_wrong_color)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = temporal_edge_labels, font_size=6)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = causal_edge_labels, font_size=6)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = subevent_edge_labels, font_size=6)

    plt.savefig(f"./images/comparison_graph_{data['doc_id'][0]}.png")
    plt.close()

def get_distance_wrong_predictions(data, temporal_pred, causal_pred, subevent_pred):
    assert len(data["temporal_labels"][0]) == len(temporal_pred)
    assert len(temporal_pred) == len(data['event_id_pairs'][0])
    temporal_wrong_distance = []
    for i, pair in enumerate(data['event_id_pairs'][0]):
        if data["temporal_labels"][0][i] != -100:
            if data["temporal_labels"][0][i] != temporal_pred[i] and data["subevent_labels"][0][i]== 6:
                temporal_wrong_distance.append(np.abs(data["mentionid2mention"][0][pair[0]]["sent_id"] - data["mentionid2mention"][0][pair[1]]["sent_id"]))

    causal_wrong_distance = []
    for i, pair in enumerate(data['event_id_pairs'][0]):
        if data["causal_labels"][0][i] != -100:
            if data["causal_labels"][0][i] != causal_pred[i] and data["subevent_labels"][0][i] == 0:
                causal_wrong_distance.append(np.abs(data["mentionid2mention"][0][pair[0]]["sent_id"] - data["mentionid2mention"][0][pair[1]]["sent_id"]))
    
    subevent_wrong_distance = []
    for i, pair in enumerate(data['event_id_pairs'][0]):
        if data["subevent_labels"][0][i] != -100:
            if data["subevent_labels"][0][i] != subevent_pred[i] and data["subevent_labels"][0][i] == 0:
                subevent_wrong_distance.append(np.abs(data["mentionid2mention"][0][pair[0]]["sent_id"] - data["mentionid2mention"][0][pair[1]]["sent_id"]))

    return temporal_wrong_distance, causal_wrong_distance, subevent_wrong_distance

def evaluate(model, dataloader, desc="", doc_number=0):
    temporal_pred_list = []
    temporal_label_list = []
    causal_pred_list = []
    causal_label_list = []
    subevent_pred_list = []
    subevent_label_list = []
    coref_train_eval_results = []
    number = 0
    temporal_wrong_distances = []
    causal_wrong_distances = []
    subevent_wrong_distances = []
    for data in tqdm(dataloader, desc=desc):
        model.eval()
        for k in data:
            if isinstance(data[k], torch.Tensor):
                data[k] = to_cuda(data[k])
        _, temporal_scores, _, _ = model(data)
        state = torch.load(os.path.join(f"./output/{args.model}/46/MAVEN-ERE", "best_COREFERENCE"))
        model.load_state_dict(state["model"])
        coref_scores, _, causal_scores, subevent_scores = model(data)
        state = torch.load(os.path.join(f"./output/{args.model}/46/MAVEN-ERE", "best_CAUSAL"))
        model.load_state_dict(state["model"])
        _, _, causal_scores, _ = model(data)
        state = torch.load(os.path.join(f"./output/{args.model}/46/MAVEN-ERE", "best_SUBEVENT"))
        model.load_state_dict(state["model"])
        _, _, _, subevent_scores = model(data)
        # coreference ###########################
        for i in range(len(coref_scores)):
            prob = coref_scores[i]
            labels = data["coref_labels"][i]
            pred_clusters, pred_event2cluster = get_predicted_clusters(prob)
            gold_event2cluster = get_event2cluster(labels)
            assert len(pred_event2cluster) == len(gold_event2cluster), print(pred_event2cluster, gold_event2cluster)
            eval_result = EvalResult(labels, gold_event2cluster, pred_clusters, pred_event2cluster)
            coref_train_eval_results.append(eval_result)
        labels = data["temporal_labels"]
        scores = temporal_scores
        scores = scores.view(-1, scores.size(-1))
        labels = labels.view(-1)
        pred = torch.argmax(scores, dim=-1)
        temporal_pred = torch.argmax(scores, dim=-1)
        temporal_pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
        temporal_label_list.extend(labels[labels>=0].cpu().numpy().tolist())
        labels = data["causal_labels"]
        scores = causal_scores
        scores = scores.view(-1, scores.size(-1))
        labels = labels.view(-1)
        pred = torch.argmax(scores, dim=-1)
        causal_pred = torch.argmax(scores, dim=-1)
        causal_pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
        causal_label_list.extend(labels[labels>=0].cpu().numpy().tolist())
        labels = data["subevent_labels"]
        scores = subevent_scores
        scores = scores.view(-1, scores.size(-1))
        labels = labels.view(-1)
        pred = torch.argmax(scores, dim=-1)
        subevent_pred = torch.argmax(scores, dim=-1)
        subevent_pred_list.extend(pred[labels>=0].cpu().numpy().tolist())
        subevent_label_list.extend(labels[labels>=0].cpu().numpy().tolist())
        #print(data)

        temporal_wrong_distance, causal_wrong_distance, subevent_wrong_distance = get_distance_wrong_predictions(data, temporal_pred, causal_pred, subevent_pred)
        temporal_wrong_distances.extend(temporal_wrong_distance)
        causal_wrong_distances.extend(causal_wrong_distance)
        subevent_wrong_distances.extend(subevent_wrong_distance)


        #if number == doc_number:
        #    draw_gold_graph(data)
        #    draw_predicted_graph(data, temporal_pred, causal_pred, subevent_pred)
        #    draw_comparison_graph(data, temporal_pred, causal_pred, subevent_pred)
        #    exit()
        #number += 1
    print(f"Temporal number: {len(temporal_wrong_distances)}")
    print(f"Overall temporal average distance: {np.nanmean(temporal_wrong_distances)}")
    print(f"Overall temporal median distance: {np.nanmedian(temporal_wrong_distances)}")
    print(f"Overall temporal distance std: {np.nanstd(temporal_wrong_distances)}")
    print(f"Causal number: {len(causal_wrong_distances)}")
    print(f"Overall causal average distance: {np.nanmean(causal_wrong_distances)}")
    print(f"Overall causal median distance: {np.nanmedian(causal_wrong_distances)}")
    print(f"Overall causal distance std: {np.nanstd(causal_wrong_distances)}")
    print(f"Subevent number: {len(subevent_wrong_distances)}")
    print(f"Overall subevent average distance: {np.nanmean(subevent_wrong_distances)}")
    print(f"Overall subevent median distance: {np.nanmedian(subevent_wrong_distances)}")
    print(f"Overall subevent distance std: {np.nanstd(subevent_wrong_distances)}")

    result_collection = {"COREFERENCE": {}}
    print("*"*20 + desc + "*"*20)
    for metric, name in zip(metrics, metric_names):
        res = evaluate_documents(coref_train_eval_results, metric)
        result_collection["COREFERENCE"][name] = {"precision": res[0], "recall": res[1], "f1": res[2]}
    print("COREFERENCE:", result_collection)
    temporal_res = classification_report(temporal_label_list, temporal_pred_list, output_dict=True, target_names=TEMP_REPORT_CLASS_NAMES, labels=TEMP_REPORT_CLASS_LABELS)
    print("TEMPORAL:", temporal_res)
    result_collection["TEMPORAL"] = temporal_res
    causal_res = classification_report(causal_label_list, causal_pred_list, output_dict=True, target_names=CAUSAL_REPORT_CLASS_NAMES, labels=CAUSAL_REPORT_CLASS_LABELS)
    print("CAUSAL:", causal_res)
    result_collection["CAUSAL"] = causal_res
    subevent_res = classification_report(subevent_label_list, subevent_pred_list, output_dict=True, target_names=SUBEVENT_REPORT_CLASS_NAMES, labels=SUBEVENT_REPORT_CLASS_LABELS)
    print("SUBEVENT:", subevent_res)
    result_collection["SUBEVENT"] = subevent_res
    return result_collection

if __name__ == "__main__":
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_steps", default=200, type=int)
    parser.add_argument("--doc_number", default=0, type=int)
    parser.add_argument("--model", default="rgcn", type=str)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--log_steps", default=50, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--bert_lr", default=2e-5, type=float)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--accumulation_steps", default=1, type=int)
    parser.add_argument("--coreference_rate", default=0.4, type=float)
    parser.add_argument("--temporal_rate", default=2.0, type=float)
    parser.add_argument("--causal_rate", default=4.0, type=float)
    parser.add_argument("--subevent_rate", default=4.0, type=float)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ignore_nonetype", action="store_true")
    parser.add_argument("--sample_rate", default=None, type=float, help="randomly sample a portion of the training data")
    args = parser.parse_args()

    TEMP_REPORT_CLASS_NAMES = [ID2TEMPREL[i] for i in range(0,len(ID2TEMPREL) - 1)]
    CAUSAL_REPORT_CLASS_NAMES = [ID2CAUSALREL[i] for i in range(1,len(ID2CAUSALREL))]
    SUBEVENT_REPORT_CLASS_NAMES = [ID2SUBEVENTREL[i] for i in range(1,len(ID2SUBEVENTREL))]

    TEMP_REPORT_CLASS_LABELS = list(range(len(ID2TEMPREL) - 1))
    CAUSAL_REPORT_CLASS_LABELS = list(range(1, len(ID2CAUSALREL)))
    SUBEVENT_REPORT_CLASS_LABELS = list(range(1, len(ID2SUBEVENTREL)))

    output_dir = Path(f"./output/analysis/{args.seed}/MAVEN-ERE")
    output_dir.mkdir(exist_ok=True, parents=True)
        
    sys.stdout = open(os.path.join(output_dir, "log.txt"), 'w')
    print(vars(args))

    set_seed(args.seed)
    
    tokenizer = RobertaTokenizer.from_pretrained("/scratch/user/kangda/MAVEN-ERE/roberta-base")
    print("loading data...")
    dev_dataloader = get_dataloader(tokenizer, "valid", max_length=256, shuffle=False, batch_size=args.batch_size, ignore_nonetype=args.ignore_nonetype)
    
    print("loading model...")
    if args.model == "rgcn":
        model = Model_RGCN(len(tokenizer))
    elif args.model == "baseline":
        model = Model(len(tokenizer))
    model = to_cuda(model)

    metrics = [b_cubed, ceafe, muc, blanc]
    metric_names = ["B-cubed", "CEAF", "MUC", "BLANC"]
    
    state = torch.load(os.path.join(f"./output/{args.model}/46/MAVEN-ERE", "best_TEMPORAL"))
    model.load_state_dict(state["model"])
    res = evaluate(model, dev_dataloader, desc="Validation", doc_number=args.doc_number)
    