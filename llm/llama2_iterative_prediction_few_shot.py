import sys
import os
from typing import List, Optional
import fire
from llama.llama import Llama, Dialog
from tqdm import tqdm
import json
from collections import defaultdict
import re
import argparse
import random
import time
import copy

def join_punctuation(seq, characters='.,;?!'):
    characters = set(characters)
    seq = iter(seq)
    current = next(seq)

    for nxt in seq:
        if nxt in characters:
            current += nxt
        else:
            yield current
            current = nxt

    yield current

def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
 
    return z

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 4096,
    max_batch_size: int = 8,
    resume_eval: bool = False,
    test: bool = False,
    shot: int = 1,
    partition: int = 0,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    docs = []
    if test: 
        output_dir = f"./output/llama2_iterative_prediction_few_shot/{shot}_shot/test"
    else:
        output_dir = f"./output/llama2_iterative_prediction_few_shot/{shot}_shot/valid"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dir = "../data/MAVEN_ERE"

    if not test:
        if os.path.exists(os.path.join(output_dir, f"log.txt")):
            os.remove(os.path.join(output_dir, f"log.txt"))
        sys.stdout = open(os.path.join(output_dir, f"log.txt"), 'w')
    else:
        if os.path.exists(os.path.join(output_dir, f"log_{partition}.txt")):
            os.remove(os.path.join(output_dir, f"log_{partition}.txt"))
        sys.stdout = open(os.path.join(output_dir, f"log_{partition}.txt"), 'w')

    resume_doc_number = 0
    if resume_eval:
        if not test:
            with open(os.path.join(output_dir, f"test_prediction.jsonl"), "r")as f:
                lines = f.readlines()
                for line in lines:
                    resume_doc_number += 1
        else:
            with open(os.path.join(output_dir, f"test_prediction_{partition}.jsonl"), "r")as f:
                lines = f.readlines()
                for line in lines:
                    resume_doc_number += 1

    all_eventnumberid2mentionid = {}
    if not test:
        with open(os.path.join(data_dir, f"valid_first10.jsonl"))as f:
            lines = f.readlines()
            for i, line in enumerate(tqdm(lines, desc="Loading test data")):

                if resume_eval and i < resume_doc_number:
                    continue
                doc = json.loads(line.strip())
                eventnumberid2mentionid = {}
                doc["event_mentions"] = []
                mention_in_sent = defaultdict(list)
                doc["corefernce_relations"] = []
                doc["coreference"] = []
                eventid2mentionid = defaultdict(list)
                mentionid2eventnumberid = {}

                eventnumberid2mention = {}

                for event in doc["events"]:
                    for mention in event["mention"]:
                        doc["event_mentions"].append(mention)
                doc["event_mentions"] = sorted(doc["event_mentions"], key=lambda x: (x["sent_id"], x["offset"][0]))
                
                id_in_event_number = 0
                for event in doc["events"]:
                    if len(event["mention"]) != 1:
                            for i in range(len(event["mention"]) - 1):
                                doc["corefernce_relations"].append([event["mention"][i]["id"], event["mention"][i+1]["id"]])

                    for mention in event["mention"]:
                        eventid2mentionid[event["id"]].append(mention["id"])
                for mention in doc["event_mentions"]:
                    num = 0
                    for previous_mention in mention_in_sent[mention["sent_id"]]:
                        if previous_mention["offset"][1] < mention["offset"][1]:
                            num += 3
                    mention["new_offset"] = [mention["offset"][0] + num, mention["offset"][1] + num + 1]
                    mention["id_in_number"] = f"Event_{id_in_event_number}"
                    mention_in_sent[mention["sent_id"]].append(mention)
                    mentionid2eventnumberid[mention["id"]] = mention["id_in_number"]
                    eventnumberid2mention[mention["id_in_number"]] = mention
                    eventnumberid2mentionid[mention["id_in_number"]] = mention["id"]
                    id_in_event_number += 1
                
                doc["TIMEX"] = sorted(doc["TIMEX"], key=lambda x: (x["sent_id"], x["offset"][0]))

                id_in_timex_number = 0 
                for mention in doc["TIMEX"]:
                    num = 0
                    for previous_mention in mention_in_sent[mention["sent_id"]]:
                        if previous_mention["offset"][1] < mention["offset"][1]:
                            num += 3
                    mention["new_offset"] = [mention["offset"][0] + num, mention["offset"][1] + num + 1]
                    mention["id_in_number"] = f"TIMEX_{id_in_timex_number}"
                    mention_in_sent[mention["sent_id"]].append(mention)
                    eventid2mentionid[mention["id"]].append(mention["id"])
                    mentionid2eventnumberid[mention["id"]] = mention["id_in_number"]
                    eventnumberid2mention[mention["id_in_number"]] = mention
                    eventnumberid2mentionid[mention["id_in_number"]] = mention["id"]
                    id_in_timex_number += 1 
                doc["eventnumberid2mention"] = eventnumberid2mention
                doc["all_mentions"] = copy.deepcopy(doc["event_mentions"])
                doc["all_mentions"].extend(copy.deepcopy(doc["TIMEX"]))
                    
                for mention in doc["event_mentions"]:
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"[")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"]")
                for mention in doc["TIMEX"]:
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"[")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"]")
                doc["sentences"] = [' '.join(join_punctuation(sentence)) for sentence in doc["tokens"]]
                doc["context"] = ' '.join(doc["sentences"])

                mentions_in_order = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', doc["context"])
                mentions_in_order_dict = {}
                for i, mention in enumerate(mentions_in_order):
                    mentions_in_order_dict[mention] = i

                temporal_order = {"BEFORE": [], "CONTAINS": [], "OVERLAP": [], "BEGINS-ON": [], "ENDS-ON": [], "SIMULTANEOUS": []}
                temporal_temp = {"BEFORE": [], "CONTAINS": [], "OVERLAP": [], "BEGINS-ON": [], "ENDS-ON": [], "SIMULTANEOUS": []}
                for temporal_type in doc["temporal_relations"]:
                    for pair in doc["temporal_relations"][temporal_type]:
                        new_pair = []
                        for event in pair:
                            new_pair.append(mentionid2eventnumberid[eventid2mentionid[event][0]])
                        temporal_temp[temporal_type].append(new_pair)
                        temporal_order[temporal_type].append(mentions_in_order_dict[new_pair[1]])
                temporal_relations = {"BEFORE": sort_list(temporal_temp["BEFORE"], temporal_order["BEFORE"]),\
                                    "CONTAINS": sort_list(temporal_temp["CONTAINS"], temporal_order["CONTAINS"]),\
                                    "OVERLAP": sort_list(temporal_temp["OVERLAP"], temporal_order["OVERLAP"]),\
                                    "BEGINS-ON": sort_list(temporal_temp["BEGINS-ON"], temporal_order["BEGINS-ON"]),\
                                    "ENDS-ON": sort_list(temporal_temp["ENDS-ON"], temporal_order["ENDS-ON"]),\
                                    "SIMULTANEOUS": sort_list(temporal_temp["SIMULTANEOUS"], temporal_order["SIMULTANEOUS"])}
                
                doc["temporal_relations"] = temporal_relations

                temporal_relations_event_order = {"BEFORE": {}, "CONTAINS": {}, "OVERLAP": {}, "BEGINS-ON": {}, "ENDS-ON": {}, "SIMULTANEOUS": {}}
                for temporal_type in doc["temporal_relations"]:
                    for pair in doc["temporal_relations"][temporal_type]:
                        if pair[1] in temporal_relations_event_order[temporal_type]:
                            temporal_relations_event_order[temporal_type][pair[1]].append(pair[0])
                        else:
                            temporal_relations_event_order[temporal_type][pair[1]] = [pair[0]]
                doc["temporal_relations_event_order"] = temporal_relations_event_order

                causal_order = {"CAUSE": [], "PRECONDITION": []}
                causal_temp = {"CAUSE": [], "PRECONDITION": []}
                for causal_type in doc["causal_relations"]:
                    for pair in doc["causal_relations"][causal_type]:
                        new_pair = []
                        for event in pair:
                            new_pair.append(mentionid2eventnumberid[eventid2mentionid[event][0]])
                        causal_temp[causal_type].append(new_pair)
                        causal_order[causal_type].append(mentions_in_order_dict[new_pair[1]])
                causal_relations = {"CAUSE": sort_list(causal_temp["CAUSE"], causal_order["CAUSE"]), \
                                    "PRECONDITION": sort_list(causal_temp["PRECONDITION"], causal_order["PRECONDITION"])}
                doc["causal_relations"] = causal_relations

                causal_relations_event_order = {"CAUSE": {}, "PRECONDITION": {}}
                for causal_type in doc["causal_relations"]:
                    for pair in doc["causal_relations"][causal_type]:
                        if pair[1] in causal_relations_event_order[causal_type]:
                            causal_relations_event_order[causal_type][pair[1]].append(pair[0])
                        else:
                            causal_relations_event_order[causal_type][pair[1]] = [pair[0]]
                doc["causal_relations_event_order"] = causal_relations_event_order
                
                subevent_temp = []
                subevent_order = []
                for pair in doc["subevent_relations"]:
                    new_pair = []
                    for event in pair:
                        new_pair.append(mentionid2eventnumberid[eventid2mentionid[event][0]])
                    subevent_temp.append(new_pair)
                    subevent_order.append(mentions_in_order_dict[new_pair[1]])
                doc["subevent_relations"] = sort_list(subevent_temp, subevent_order)

                subevent_relations_event_order = {}
                for pair in doc["subevent_relations"]:
                    if pair[1] in subevent_relations_event_order:
                        subevent_relations_event_order[pair[1]].append(pair[0])
                    else:
                        subevent_relations_event_order[pair[1]] = [pair[0]]
                doc["subevent_relations_event_order"] = subevent_relations_event_order

                coref_temp = []
                coref_order = []
                for pair in doc["corefernce_relations"]:
                    new_pair = []
                    for event in pair:
                        new_pair.append(mentionid2eventnumberid[event])
                    coref_temp.append(new_pair)
                    coref_order.append(mentions_in_order_dict[new_pair[1]])
                doc["coreference_relations"] = sort_list(coref_temp, coref_order)

                
                coreference_relations_event_order = {}
                for i, event in enumerate(doc["events"]):
                    for j, mention in enumerate(event["mention"]):
                        coreference_relations_event_order[mentionid2eventnumberid[mention["id"]]] = []
                        for k, mention_k in enumerate(event["mention"]):
                            if k != j:
                                coreference_relations_event_order[mentionid2eventnumberid[mention["id"]]].append(mentionid2eventnumberid[mention_k["id"]]) 
                doc["coreference_relations_event_order"] = coreference_relations_event_order

                doc["coreference"] = []
                for event in doc["events"]:
                    coref = []
                    for mention in event["mention"]:
                        coref.append(mentionid2eventnumberid[mention["id"]])
                    doc["coreference"].append(coref)
                doc["coreference"] = sorted(doc["coreference"], key=lambda x: -len(x))

                doc["coref_context_inputs"] = []
                doc["coref_answer_inputs"] = []
                mentions_appeared = []
                cluster_appeared = []
                coref_tokens = copy.deepcopy(doc["tokens"])
                for i, sentence in enumerate(doc["sentences"]):
                    mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)
                    coref_answer_input = ""
                    num = 0
                    
                    for mention in mentions_in_sentence:
                        done = False
                        for mention_appeared in mentions_appeared:
                            for j, cluster in enumerate(doc["coreference"]):
                                if mention in cluster and mention_appeared in cluster:
                                    if j in cluster_appeared:
                                        if done: continue
                                        num += 1
                                        coref_answer_input += f"{mention} COREFERENCE [ {j}; "
                                        coref_tokens[i].insert(eventnumberid2mention[mention]["new_offset"][0]+ num, str(j))
                                        done = True
                                        continue
                                    num += 1
                                    coref_answer_input += f"{mention} COREFERENCE {mention_appeared}; "
                                    coref_tokens[i].insert(eventnumberid2mention[mention]["new_offset"][0]+num, str(j))
                                    coref_tokens[eventnumberid2mention[mention_appeared]["sent_id"]].insert(eventnumberid2mention[mention_appeared]["new_offset"][0] + 1, str(j))
                                    cluster_appeared.append(j)
                        mentions_appeared.append(mention)
                    accumulated_context = [' '.join(join_punctuation(coref_tokens[index])) for index in range(i)]
                    accumulated_context.append(sentence)
                    accumulated_context = ' '.join(join_punctuation(accumulated_context))
                    coref_answer_input += "SHIFT;"
                    #print(accumulated_context)
                    #print(coref_answer_input)
                    doc["coref_answer_inputs"].append(coref_answer_input)
                    doc["coref_context_inputs"].append(accumulated_context)

                doc["temporal_context_inputs"] = []
                doc["temporal_answer_inputs"] = []
                temporal_tokens = copy.deepcopy(doc["tokens"])
                mentions_appeared = {}
                mentions_inserted = []
                eventnumberid2appearedrelations = {"BEFORE": {}, "CONTAINS": {}, "OVERLAP": {}, "BEGINS-ON": {}, "ENDS-ON": {}, "SIMULTANEOUS": {}}
                for i, sentence in enumerate(doc["sentences"]):
                    mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)
                    temporal_answer_input = ""
                    for mention in mentions_in_sentence:
                        done = False
                        for mention_appeared in mentions_appeared:
                            for temporal_type in ["BEFORE", "CONTAINS", "OVERLAP", "BEGINS-ON", "ENDS-ON", "SIMULTANEOUS"]:
                                for pair in doc["temporal_relations"][temporal_type]:
                                    if mention == pair[0] and mention_appeared == pair[1]:
                                        temporal_answer_input += f"{mention} {temporal_type} {mention_appeared}; "
                                        if pair[1] in eventnumberid2appearedrelations[temporal_type]:
                                            eventnumberid2appearedrelations[temporal_type][pair[1]].append(f"{mention} {temporal_type} {mention_appeared}")
                                        else:
                                            eventnumberid2appearedrelations[temporal_type][pair[1]] = [f"{mention} {temporal_type} {mention_appeared}"]
                                    elif mention == pair[1] and mention_appeared == pair[0]:
                                        temporal_answer_input += f"{mention_appeared} {temporal_type} {mention}; "
                                        if pair[1] in eventnumberid2appearedrelations[temporal_type]:
                                            eventnumberid2appearedrelations[temporal_type][pair[1]].append(f"{mention_appeared} {temporal_type} {mention}")
                                        else:
                                            eventnumberid2appearedrelations[temporal_type][pair[1]] = [f"{mention_appeared} {temporal_type} {mention}"]
                        mentions_appeared[mention] = len(mentions_in_sentence)

                    num = 0
                    for mention_appeared in mentions_appeared:
                        if mentions_appeared[mention_appeared] == num:
                            num = 0
                        num += 1
                        relations_for_inserting = ""
                        for temporal_type in ["BEFORE", "CONTAINS", "OVERLAP", "BEGINS-ON", "ENDS-ON", "SIMULTANEOUS"]:
                            if mention_appeared in eventnumberid2appearedrelations[temporal_type]:
                                relations_for_inserting += ';'.join(eventnumberid2appearedrelations[temporal_type][mention_appeared])
                                relations_for_inserting += ';'
                        
                        if mention_appeared in mentions_inserted:
                            temporal_tokens[eventnumberid2mention[mention_appeared]["sent_id"]][eventnumberid2mention[mention_appeared]["new_offset"][1] + num] = relations_for_inserting
                        else:
                            temporal_tokens[eventnumberid2mention[mention_appeared]["sent_id"]].insert(eventnumberid2mention[mention_appeared]["new_offset"][1] + num, relations_for_inserting)
                            mentions_inserted.append(mention_appeared)
                        
                        if mentions_appeared[mention_appeared] == num:
                            num = 0
                    accumulated_context = [' '.join(join_punctuation(temporal_tokens[index])) for index in range(i)]
                    accumulated_context.append(sentence)
                    accumulated_context = ' '.join(join_punctuation(accumulated_context))
                    temporal_answer_input += "SHIFT;"
                    #print(accumulated_context)
                    #print(temporal_answer_input)
                    doc["temporal_answer_inputs"].append(temporal_answer_input)
                    doc["temporal_context_inputs"].append(accumulated_context)

                doc["causal_context_inputs"] = []
                doc["causal_answer_inputs"] = []
                causal_tokens = copy.deepcopy(doc["tokens"])
                mentions_appeared = {}
                mentions_inserted = []
                eventnumberid2appearedrelations = {"CAUSE": {}, "PRECONDITION": {}}
                for i, sentence in enumerate(doc["sentences"]):
                    mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)
                    causal_answer_input = ""
                    for mention in mentions_in_sentence:
                        done = False
                        for mention_appeared in mentions_appeared:
                            for causal_type in ["CAUSE", "PRECONDITION"]:
                                for pair in doc["causal_relations"][causal_type]:
                                    if mention == pair[0] and mention_appeared == pair[1]:
                                        causal_answer_input += f"{mention} {causal_type} {mention_appeared}; "
                                        if pair[1] in eventnumberid2appearedrelations[causal_type]:
                                            eventnumberid2appearedrelations[causal_type][pair[1]].append(f"{mention} {causal_type} {mention_appeared}")
                                        else:
                                            eventnumberid2appearedrelations[causal_type][pair[1]] = [f"{mention} {causal_type} {mention_appeared}"]
                                    elif mention == pair[1] and mention_appeared == pair[0]:
                                        causal_answer_input += f"{mention_appeared} {causal_type} {mention}; "
                                        if pair[1] in eventnumberid2appearedrelations[causal_type]:
                                            eventnumberid2appearedrelations[causal_type][pair[1]].append(f"{mention_appeared} {causal_type} {mention}")
                                        else:
                                            eventnumberid2appearedrelations[causal_type][pair[1]] = [f"{mention_appeared} {causal_type} {mention}"]
                        mentions_appeared[mention] = len(mentions_in_sentence)

                    num = 0
                    for mention_appeared in mentions_appeared:
                        if mentions_appeared[mention_appeared] == num:
                            num = 0
                        num += 1
                        relations_for_inserting = ""
                        for causal_type in ["CAUSE", "PRECONDITION"]:
                            if mention_appeared in eventnumberid2appearedrelations[causal_type]:
                                relations_for_inserting += ';'.join(eventnumberid2appearedrelations[causal_type][mention_appeared])
                                relations_for_inserting += ';'
                        
                        if mention_appeared in mentions_inserted:
                            causal_tokens[eventnumberid2mention[mention_appeared]["sent_id"]][eventnumberid2mention[mention_appeared]["new_offset"][1] + num] = relations_for_inserting
                        else:
                            causal_tokens[eventnumberid2mention[mention_appeared]["sent_id"]].insert(eventnumberid2mention[mention_appeared]["new_offset"][1] + num, relations_for_inserting)
                            mentions_inserted.append(mention_appeared)
                        
                        if mentions_appeared[mention_appeared] == num:
                            num = 0
                    accumulated_context = [' '.join(join_punctuation(causal_tokens[index])) for index in range(i)]
                    accumulated_context.append(sentence)
                    accumulated_context = ' '.join(join_punctuation(accumulated_context))
                    causal_answer_input += "SHIFT;"
                    #print(accumulated_context)
                    #print(causal_answer_input)
                    doc["causal_answer_inputs"].append(causal_answer_input)
                    doc["causal_context_inputs"].append(accumulated_context)

                doc["subevent_context_inputs"] = []
                doc["subevent_answer_inputs"] = []
                subevent_tokens = copy.deepcopy(doc["tokens"])
                mentions_appeared = {}
                mentions_inserted = []
                eventnumberid2appearedrelations = {"SUBEVENT": {}}
                for i, sentence in enumerate(doc["sentences"]):
                    mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)
                    subevent_answer_input = ""
                    for mention in mentions_in_sentence:
                        done = False
                        for mention_appeared in mentions_appeared:
                            for subevent_type in ["SUBEVENT"]:
                                for pair in doc["subevent_relations"]:
                                    if mention == pair[0] and mention_appeared == pair[1]:
                                        subevent_answer_input += f"{mention} {subevent_type} {mention_appeared}; "
                                        if pair[1] in eventnumberid2appearedrelations[subevent_type]:
                                            eventnumberid2appearedrelations[subevent_type][pair[1]].append(f"{mention} {subevent_type} {mention_appeared}")
                                        else:
                                            eventnumberid2appearedrelations[subevent_type][pair[1]] = [f"{mention} {subevent_type} {mention_appeared}"]
                                    elif mention == pair[1] and mention_appeared == pair[0]:
                                        subevent_answer_input += f"{mention_appeared} {subevent_type} {mention}; "
                                        if pair[1] in eventnumberid2appearedrelations[subevent_type]:
                                            eventnumberid2appearedrelations[subevent_type][pair[1]].append(f"{mention_appeared} {subevent_type} {mention}")
                                        else:
                                            eventnumberid2appearedrelations[subevent_type][pair[1]] = [f"{mention_appeared} {subevent_type} {mention}"]
                        mentions_appeared[mention] = len(mentions_in_sentence)

                    num = 0
                    for mention_appeared in mentions_appeared:
                        if mentions_appeared[mention_appeared] == num:
                            num = 0
                        num += 1
                        relations_for_inserting = ""
                        for subevent_type in ["SUBEVENT"]:
                            if mention_appeared in eventnumberid2appearedrelations[subevent_type]:
                                relations_for_inserting += ';'.join(eventnumberid2appearedrelations[subevent_type][mention_appeared])
                                relations_for_inserting += ';'
                        
                        if mention_appeared in mentions_inserted:
                            subevent_tokens[eventnumberid2mention[mention_appeared]["sent_id"]][eventnumberid2mention[mention_appeared]["new_offset"][1] + num] = relations_for_inserting
                        else:
                            subevent_tokens[eventnumberid2mention[mention_appeared]["sent_id"]].insert(eventnumberid2mention[mention_appeared]["new_offset"][1] + num, relations_for_inserting)
                            mentions_inserted.append(mention_appeared)
                        
                        if mentions_appeared[mention_appeared] == num:
                            num = 0
                    accumulated_context = [' '.join(join_punctuation(subevent_tokens[index])) for index in range(i)]
                    accumulated_context.append(sentence)
                    accumulated_context = ' '.join(join_punctuation(accumulated_context))
                    subevent_answer_input += "SHIFT;"
                    #print(accumulated_context)
                    #print(subevent_answer_input)
                    doc["subevent_answer_inputs"].append(subevent_answer_input)
                    doc["subevent_context_inputs"].append(accumulated_context)
                all_eventnumberid2mentionid[doc["id"]] = eventnumberid2mentionid
                docs.append(doc)
                #print(doc)
    else:
        with open(os.path.join(data_dir, f"test.jsonl"))as f:
            lines = f.readlines()
            batch_size = int(len(lines) / 10) + 1
            start = partition * batch_size
            end = start + batch_size if partition != 9 else len(lines)
            print(start, end)
            for i, line in enumerate(tqdm(lines[start:end], desc="Loading test data")):

                if resume_eval and i < resume_doc_number:
                    continue
                doc = json.loads(line.strip())
                eventnumberid2mentionid = {}
                mention_in_sent = defaultdict(list)
                #eventid2mentionid = defaultdict(list)
                mentionid2eventnumberid = {}

                eventnumberid2mention = {}

                doc["event_mentions"] = sorted(doc["event_mentions"], key=lambda x: (x["sent_id"], x["offset"][0]))
                
                id_in_event_number = 0
                for mention in doc["event_mentions"]:
                    num = 0
                    for previous_mention in mention_in_sent[mention["sent_id"]]:
                        if previous_mention["offset"][1] < mention["offset"][1]:
                            num += 3
                    mention["new_offset"] = [mention["offset"][0] + num, mention["offset"][1] + num + 1]
                    mention["id_in_number"] = f"Event_{id_in_event_number}"
                    mention_in_sent[mention["sent_id"]].append(mention)
                    mentionid2eventnumberid[mention["id"]] = mention["id_in_number"]
                    eventnumberid2mention[mention["id_in_number"]] = mention
                    eventnumberid2mentionid[mention["id_in_number"]] = mention["id"]
                    id_in_event_number += 1
                
                doc["TIMEX"] = sorted(doc["TIMEX"], key=lambda x: (x["sent_id"], x["offset"][0]))

                id_in_timex_number = 0 
                for mention in doc["TIMEX"]:
                    num = 0
                    for previous_mention in mention_in_sent[mention["sent_id"]]:
                        if previous_mention["offset"][1] < mention["offset"][1]:
                            num += 3
                    mention["new_offset"] = [mention["offset"][0] + num, mention["offset"][1] + num + 1]
                    mention["id_in_number"] = f"TIMEX_{id_in_timex_number}"
                    mention_in_sent[mention["sent_id"]].append(mention)
                    mentionid2eventnumberid[mention["id"]] = mention["id_in_number"]
                    eventnumberid2mention[mention["id_in_number"]] = mention
                    eventnumberid2mentionid[mention["id_in_number"]] = mention["id"]
                    id_in_timex_number += 1 
                doc["eventnumberid2mention"] = eventnumberid2mention
                doc["all_mentions"] = copy.deepcopy(doc["event_mentions"])
                doc["all_mentions"].extend(copy.deepcopy(doc["TIMEX"]))
                    
                for mention in doc["event_mentions"]:
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"[")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"]")
                for mention in doc["TIMEX"]:
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"[")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"]")
                doc["sentences"] = [' '.join(join_punctuation(sentence)) for sentence in doc["tokens"]]
                doc["context"] = ' '.join(doc["sentences"])

                print(doc)
                all_eventnumberid2mentionid[doc["id"]] = eventnumberid2mentionid
                docs.append(doc)
        #exit()
    if not resume_eval:
        with open(os.path.join(output_dir, f"eventnumberid2mentionid_{partition}.jsonl"), "w") as outfile:
            json.dump(all_eventnumberid2mentionid, outfile)

    example_docs = []
    with open(f"./example_few_shot.jsonl")as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines, desc="Loading example data")):
            example_doc = json.loads(line.strip())
            example_doc["event_mentions"] = []
            mention_in_sent = defaultdict(list)
            example_doc["corefernce_relations"] = []
            eventid2mentionid = defaultdict(list)
            mentionid2eventnumberid = {}
            eventnumberid2mention = {}

            for event in example_doc["events"]:
                for mention in event["mention"]:
                    example_doc["event_mentions"].append(mention)
            example_doc["event_mentions"] = sorted(example_doc["event_mentions"], key=lambda x: (x["sent_id"], x["offset"][0]))
            
            id_in_event_number = 0
            for event in example_doc["events"]:
                if len(event["mention"]) != 1:
                        for i in range(len(event["mention"]) - 1):
                            example_doc["corefernce_relations"].append([event["mention"][i]["id"], event["mention"][i+1]["id"]])

                for mention in event["mention"]:
                    eventid2mentionid[event["id"]].append(mention["id"])
            for mention in example_doc["event_mentions"]:
                num = 0
                for previous_mention in mention_in_sent[mention["sent_id"]]:
                    if previous_mention["offset"][1] < mention["offset"][1]:
                        num += 3
                mention["new_offset"] = [mention["offset"][0] + num, mention["offset"][1] + num + 1]
                mention["id_in_number"] = f"Event_{id_in_event_number}"
                mention_in_sent[mention["sent_id"]].append(mention)
                mentionid2eventnumberid[mention["id"]] = mention["id_in_number"]
                eventnumberid2mention[mention["id_in_number"]] = mention
                id_in_event_number += 1
            
            example_doc["TIMEX"] = sorted(example_doc["TIMEX"], key=lambda x: (x["sent_id"], x["offset"][0]))

            id_in_timex_number = 0 
            for mention in example_doc["TIMEX"]:
                num = 0
                for previous_mention in mention_in_sent[mention["sent_id"]]:
                    if previous_mention["offset"][1] < mention["offset"][1]:
                        num += 3
                mention["new_offset"] = [mention["offset"][0] + num, mention["offset"][1] + num + 1]
                mention["id_in_number"] = f"TIMEX_{id_in_timex_number}"
                mention_in_sent[mention["sent_id"]].append(mention)
                eventid2mentionid[mention["id"]].append(mention["id"])
                mentionid2eventnumberid[mention["id"]] = mention["id_in_number"]
                eventnumberid2mention[mention["id_in_number"]] = mention
                id_in_timex_number += 1 

            example_doc["all_mentions"] = copy.deepcopy(example_doc["event_mentions"])
            example_doc["all_mentions"].extend(copy.deepcopy(example_doc["TIMEX"]))
                
            for mention in example_doc["event_mentions"]:
                example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"[")
                example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"]")
            for mention in example_doc["TIMEX"]:
                example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"[")
                example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"]")
            example_doc["sentences"] = [' '.join(join_punctuation(sentence)) for sentence in example_doc["tokens"]]
            example_doc["context"] = ' '.join(example_doc["sentences"])

            mentions_in_order = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', example_doc["context"])
            mentions_in_order_dict = {}
            for i, mention in enumerate(mentions_in_order):
                mentions_in_order_dict[mention] = i

            temporal_order = {"BEFORE": [], "CONTAINS": [], "OVERLAP": [], "BEGINS-ON": [], "ENDS-ON": [], "SIMULTANEOUS": []}
            temporal_temp = {"BEFORE": [], "CONTAINS": [], "OVERLAP": [], "BEGINS-ON": [], "ENDS-ON": [], "SIMULTANEOUS": []}
            for temporal_type in example_doc["temporal_relations"]:
                for pair in example_doc["temporal_relations"][temporal_type]:
                    new_pair = []
                    for event in pair:
                        new_pair.append(mentionid2eventnumberid[eventid2mentionid[event][0]])
                    temporal_temp[temporal_type].append(new_pair)
                    temporal_order[temporal_type].append(mentions_in_order_dict[new_pair[1]])
            temporal_relations = {"BEFORE": sort_list(temporal_temp["BEFORE"], temporal_order["BEFORE"]),\
                                "CONTAINS": sort_list(temporal_temp["CONTAINS"], temporal_order["CONTAINS"]),\
                                "OVERLAP": sort_list(temporal_temp["OVERLAP"], temporal_order["OVERLAP"]),\
                                "BEGINS-ON": sort_list(temporal_temp["BEGINS-ON"], temporal_order["BEGINS-ON"]),\
                                "ENDS-ON": sort_list(temporal_temp["ENDS-ON"], temporal_order["ENDS-ON"]),\
                                "SIMULTANEOUS": sort_list(temporal_temp["SIMULTANEOUS"], temporal_order["SIMULTANEOUS"])}
            
            example_doc["temporal_relations"] = temporal_relations

            temporal_relations_event_order = {"BEFORE": {}, "CONTAINS": {}, "OVERLAP": {}, "BEGINS-ON": {}, "ENDS-ON": {}, "SIMULTANEOUS": {}}
            for temporal_type in example_doc["temporal_relations"]:
                for pair in example_doc["temporal_relations"][temporal_type]:
                    if pair[1] in temporal_relations_event_order[temporal_type]:
                        temporal_relations_event_order[temporal_type][pair[1]].append(pair[0])
                    else:
                        temporal_relations_event_order[temporal_type][pair[1]] = [pair[0]]
            example_doc["temporal_relations_event_order"] = temporal_relations_event_order

            causal_order = {"CAUSE": [], "PRECONDITION": []}
            causal_temp = {"CAUSE": [], "PRECONDITION": []}
            for causal_type in example_doc["causal_relations"]:
                for pair in example_doc["causal_relations"][causal_type]:
                    new_pair = []
                    for event in pair:
                        new_pair.append(mentionid2eventnumberid[eventid2mentionid[event][0]])
                    causal_temp[causal_type].append(new_pair)
                    causal_order[causal_type].append(mentions_in_order_dict[new_pair[1]])
            causal_relations = {"CAUSE": sort_list(causal_temp["CAUSE"], causal_order["CAUSE"]), \
                                "PRECONDITION": sort_list(causal_temp["PRECONDITION"], causal_order["PRECONDITION"])}
            example_doc["causal_relations"] = causal_relations

            causal_relations_event_order = {"CAUSE": {}, "PRECONDITION": {}}
            for causal_type in example_doc["causal_relations"]:
                for pair in example_doc["causal_relations"][causal_type]:
                    if pair[1] in causal_relations_event_order[causal_type]:
                        causal_relations_event_order[causal_type][pair[1]].append(pair[0])
                    else:
                        causal_relations_event_order[causal_type][pair[1]] = [pair[0]]
            example_doc["causal_relations_event_order"] = causal_relations_event_order
            
            subevent_temp = []
            subevent_order = []
            for pair in example_doc["subevent_relations"]:
                new_pair = []
                for event in pair:
                    new_pair.append(mentionid2eventnumberid[eventid2mentionid[event][0]])
                subevent_temp.append(new_pair)
                subevent_order.append(mentions_in_order_dict[new_pair[1]])
            example_doc["subevent_relations"] = sort_list(subevent_temp, subevent_order)

            subevent_relations_event_order = {}
            for pair in example_doc["subevent_relations"]:
                if pair[1] in subevent_relations_event_order:
                    subevent_relations_event_order[pair[1]].append(pair[0])
                else:
                    subevent_relations_event_order[pair[1]] = [pair[0]]
            example_doc["subevent_relations_event_order"] = subevent_relations_event_order

            coref_temp = []
            coref_order = []
            for pair in example_doc["corefernce_relations"]:
                new_pair = []
                for event in pair:
                    new_pair.append(mentionid2eventnumberid[event])
                coref_temp.append(new_pair)
                coref_order.append(mentions_in_order_dict[new_pair[1]])
            example_doc["coreference_relations"] = sort_list(coref_temp, coref_order)

            
            coreference_relations_event_order = {}
            for i, event in enumerate(example_doc["events"]):
                for j, mention in enumerate(event["mention"]):
                    coreference_relations_event_order[mentionid2eventnumberid[mention["id"]]] = []
                    for k, mention_k in enumerate(event["mention"]):
                        if k != j:
                            coreference_relations_event_order[mentionid2eventnumberid[mention["id"]]].append(mentionid2eventnumberid[mention_k["id"]]) 
            example_doc["coreference_relations_event_order"] = coreference_relations_event_order

            example_doc["coreference"] = []
            for event in example_doc["events"]:
                coref = []
                for mention in event["mention"]:
                    coref.append(mentionid2eventnumberid[mention["id"]])
                example_doc["coreference"].append(coref)
            example_doc["coreference"] = sorted(example_doc["coreference"], key=lambda x: -len(x))

            example_doc["coref_context_inputs"] = []
            example_doc["coref_answer_inputs"] = []
            mentions_appeared = []
            cluster_appeared = []
            coref_tokens = copy.deepcopy(example_doc["tokens"])
            for i, sentence in enumerate(example_doc["sentences"]):
                mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)
                coref_answer_input = ""
                num = 0
                
                for mention in mentions_in_sentence:
                    done = False
                    for mention_appeared in mentions_appeared:
                        for j, cluster in enumerate(example_doc["coreference"]):
                            if mention in cluster and mention_appeared in cluster:
                                if j in cluster_appeared:
                                    if done: continue
                                    num += 1
                                    coref_answer_input += f"{mention} COREFERENCE [ {j}; "
                                    coref_tokens[i].insert(eventnumberid2mention[mention]["new_offset"][0]+ num, str(j))
                                    done = True
                                    continue
                                num += 1
                                coref_answer_input += f"{mention} COREFERENCE {mention_appeared}; "
                                coref_tokens[i].insert(eventnumberid2mention[mention]["new_offset"][0]+num, str(j))
                                coref_tokens[eventnumberid2mention[mention_appeared]["sent_id"]].insert(eventnumberid2mention[mention_appeared]["new_offset"][0] + 1, str(j))
                                cluster_appeared.append(j)
                    mentions_appeared.append(mention)
                accumulated_context = [' '.join(join_punctuation(coref_tokens[index])) for index in range(i)]
                accumulated_context.append(sentence)
                accumulated_context = ' '.join(join_punctuation(accumulated_context))
                coref_answer_input += "SHIFT;"
                #print(accumulated_context)
                #print(coref_answer_input)
                example_doc["coref_answer_inputs"].append(coref_answer_input)
                example_doc["coref_context_inputs"].append(accumulated_context)

            example_doc["temporal_context_inputs"] = []
            example_doc["temporal_answer_inputs"] = []
            temporal_tokens = copy.deepcopy(example_doc["tokens"])
            mentions_appeared = {}
            mentions_inserted = []
            eventnumberid2appearedrelations = {"BEFORE": {}, "CONTAINS": {}, "OVERLAP": {}, "BEGINS-ON": {}, "ENDS-ON": {}, "SIMULTANEOUS": {}}
            for i, sentence in enumerate(example_doc["sentences"]):
                mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)
                temporal_answer_input = ""
                for mention in mentions_in_sentence:
                    done = False
                    for mention_appeared in mentions_appeared:
                        for temporal_type in ["BEFORE", "CONTAINS", "OVERLAP", "BEGINS-ON", "ENDS-ON", "SIMULTANEOUS"]:
                            for pair in example_doc["temporal_relations"][temporal_type]:
                                if mention == pair[0] and mention_appeared == pair[1]:
                                    temporal_answer_input += f"{mention} {temporal_type} {mention_appeared}; "
                                    if pair[1] in eventnumberid2appearedrelations[temporal_type]:
                                        eventnumberid2appearedrelations[temporal_type][pair[1]].append(f"{mention} {temporal_type} {mention_appeared}")
                                    else:
                                        eventnumberid2appearedrelations[temporal_type][pair[1]] = [f"{mention} {temporal_type} {mention_appeared}"]
                                elif mention == pair[1] and mention_appeared == pair[0]:
                                    temporal_answer_input += f"{mention_appeared} {temporal_type} {mention}; "
                                    if pair[1] in eventnumberid2appearedrelations[temporal_type]:
                                        eventnumberid2appearedrelations[temporal_type][pair[1]].append(f"{mention_appeared} {temporal_type} {mention}")
                                    else:
                                        eventnumberid2appearedrelations[temporal_type][pair[1]] = [f"{mention_appeared} {temporal_type} {mention}"]
                    mentions_appeared[mention] = len(mentions_in_sentence)

                num = 0
                for mention_appeared in mentions_appeared:
                    if mentions_appeared[mention_appeared] == num:
                        num = 0
                    num += 1
                    relations_for_inserting = ""
                    for temporal_type in ["BEFORE", "CONTAINS", "OVERLAP", "BEGINS-ON", "ENDS-ON", "SIMULTANEOUS"]:
                        if mention_appeared in eventnumberid2appearedrelations[temporal_type]:
                            relations_for_inserting += ';'.join(eventnumberid2appearedrelations[temporal_type][mention_appeared])
                            relations_for_inserting += ';'
                    
                    if mention_appeared in mentions_inserted:
                        temporal_tokens[eventnumberid2mention[mention_appeared]["sent_id"]][eventnumberid2mention[mention_appeared]["new_offset"][1] + num] = relations_for_inserting
                    else:
                        temporal_tokens[eventnumberid2mention[mention_appeared]["sent_id"]].insert(eventnumberid2mention[mention_appeared]["new_offset"][1] + num, relations_for_inserting)
                        mentions_inserted.append(mention_appeared)
                    
                    if mentions_appeared[mention_appeared] == num:
                        num = 0
                accumulated_context = [' '.join(join_punctuation(temporal_tokens[index])) for index in range(i)]
                accumulated_context.append(sentence)
                accumulated_context = ' '.join(join_punctuation(accumulated_context))
                temporal_answer_input += "SHIFT;"
                #print(accumulated_context)
                #print(temporal_answer_input)
                example_doc["temporal_answer_inputs"].append(temporal_answer_input)
                example_doc["temporal_context_inputs"].append(accumulated_context)

            example_doc["causal_context_inputs"] = []
            example_doc["causal_answer_inputs"] = []
            causal_tokens = copy.deepcopy(example_doc["tokens"])
            mentions_appeared = {}
            mentions_inserted = []
            eventnumberid2appearedrelations = {"CAUSE": {}, "PRECONDITION": {}}
            for i, sentence in enumerate(example_doc["sentences"]):
                mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)
                causal_answer_input = ""
                for mention in mentions_in_sentence:
                    done = False
                    for mention_appeared in mentions_appeared:
                        for causal_type in ["CAUSE", "PRECONDITION"]:
                            for pair in example_doc["causal_relations"][causal_type]:
                                if mention == pair[0] and mention_appeared == pair[1]:
                                    causal_answer_input += f"{mention} {causal_type} {mention_appeared}; "
                                    if pair[1] in eventnumberid2appearedrelations[causal_type]:
                                        eventnumberid2appearedrelations[causal_type][pair[1]].append(f"{mention} {causal_type} {mention_appeared}")
                                    else:
                                        eventnumberid2appearedrelations[causal_type][pair[1]] = [f"{mention} {causal_type} {mention_appeared}"]
                                elif mention == pair[1] and mention_appeared == pair[0]:
                                    causal_answer_input += f"{mention_appeared} {causal_type} {mention}; "
                                    if pair[1] in eventnumberid2appearedrelations[causal_type]:
                                        eventnumberid2appearedrelations[causal_type][pair[1]].append(f"{mention_appeared} {causal_type} {mention}")
                                    else:
                                        eventnumberid2appearedrelations[causal_type][pair[1]] = [f"{mention_appeared} {causal_type} {mention}"]
                    mentions_appeared[mention] = len(mentions_in_sentence)

                num = 0
                for mention_appeared in mentions_appeared:
                    if mentions_appeared[mention_appeared] == num:
                        num = 0
                    num += 1
                    relations_for_inserting = ""
                    for causal_type in ["CAUSE", "PRECONDITION"]:
                        if mention_appeared in eventnumberid2appearedrelations[causal_type]:
                            relations_for_inserting += ';'.join(eventnumberid2appearedrelations[causal_type][mention_appeared])
                            relations_for_inserting += ';'
                    
                    if mention_appeared in mentions_inserted:
                        causal_tokens[eventnumberid2mention[mention_appeared]["sent_id"]][eventnumberid2mention[mention_appeared]["new_offset"][1] + num] = relations_for_inserting
                    else:
                        causal_tokens[eventnumberid2mention[mention_appeared]["sent_id"]].insert(eventnumberid2mention[mention_appeared]["new_offset"][1] + num, relations_for_inserting)
                        mentions_inserted.append(mention_appeared)
                    
                    if mentions_appeared[mention_appeared] == num:
                        num = 0
                accumulated_context = [' '.join(join_punctuation(causal_tokens[index])) for index in range(i)]
                accumulated_context.append(sentence)
                accumulated_context = ' '.join(join_punctuation(accumulated_context))
                causal_answer_input += "SHIFT;"
                #print(accumulated_context)
                #print(causal_answer_input)
                example_doc["causal_answer_inputs"].append(causal_answer_input)
                example_doc["causal_context_inputs"].append(accumulated_context)

            example_doc["subevent_context_inputs"] = []
            example_doc["subevent_answer_inputs"] = []
            subevent_tokens = copy.deepcopy(example_doc["tokens"])
            mentions_appeared = {}
            mentions_inserted = []
            eventnumberid2appearedrelations = {"SUBEVENT": {}}
            for i, sentence in enumerate(example_doc["sentences"]):
                mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)
                subevent_answer_input = ""
                for mention in mentions_in_sentence:
                    done = False
                    for mention_appeared in mentions_appeared:
                        for subevent_type in ["SUBEVENT"]:
                            for pair in example_doc["subevent_relations"]:
                                if mention == pair[0] and mention_appeared == pair[1]:
                                    subevent_answer_input += f"{mention} {subevent_type} {mention_appeared}; "
                                    if pair[1] in eventnumberid2appearedrelations[subevent_type]:
                                        eventnumberid2appearedrelations[subevent_type][pair[1]].append(f"{mention} {subevent_type} {mention_appeared}")
                                    else:
                                        eventnumberid2appearedrelations[subevent_type][pair[1]] = [f"{mention} {subevent_type} {mention_appeared}"]
                                elif mention == pair[1] and mention_appeared == pair[0]:
                                    subevent_answer_input += f"{mention_appeared} {subevent_type} {mention}; "
                                    if pair[1] in eventnumberid2appearedrelations[subevent_type]:
                                        eventnumberid2appearedrelations[subevent_type][pair[1]].append(f"{mention_appeared} {subevent_type} {mention}")
                                    else:
                                        eventnumberid2appearedrelations[subevent_type][pair[1]] = [f"{mention_appeared} {subevent_type} {mention}"]
                    mentions_appeared[mention] = len(mentions_in_sentence)

                num = 0
                for mention_appeared in mentions_appeared:
                    if mentions_appeared[mention_appeared] == num:
                        num = 0
                    num += 1
                    relations_for_inserting = ""
                    for subevent_type in ["SUBEVENT"]:
                        if mention_appeared in eventnumberid2appearedrelations[subevent_type]:
                            relations_for_inserting += ';'.join(eventnumberid2appearedrelations[subevent_type][mention_appeared])
                            relations_for_inserting += ';'
                    
                    if mention_appeared in mentions_inserted:
                        subevent_tokens[eventnumberid2mention[mention_appeared]["sent_id"]][eventnumberid2mention[mention_appeared]["new_offset"][1] + num] = relations_for_inserting
                    else:
                        subevent_tokens[eventnumberid2mention[mention_appeared]["sent_id"]].insert(eventnumberid2mention[mention_appeared]["new_offset"][1] + num, relations_for_inserting)
                        mentions_inserted.append(mention_appeared)
                    
                    if mentions_appeared[mention_appeared] == num:
                        num = 0
                accumulated_context = [' '.join(join_punctuation(subevent_tokens[index])) for index in range(i)]
                accumulated_context.append(sentence)
                accumulated_context = ' '.join(join_punctuation(accumulated_context))
                subevent_answer_input += "SHIFT;"
                #print(accumulated_context)
                #print(subevent_answer_input)
                example_doc["subevent_answer_inputs"].append(subevent_answer_input)
                example_doc["subevent_context_inputs"].append(accumulated_context)
            example_docs.append(example_doc)
            #print(example_doc)
        #print(example_docs)
        #exit()

    for i, doc in enumerate(tqdm(docs, desc="Predicting")):
        #if i == 1:
        #    exit()

        print(doc["id"])
        if  doc["id"] == "d791d9d612faaaf9ca63e206aee82489" or \
            doc["id"] == "d8ec928eebf398848c96f7870385d616" or \
            doc["id"] == "6b2e8c050e30872e49c2f46edb4ac044" or \
            doc["id"] == "ad6719cf84c9d62f57045a26f344e56c" or \
            doc["id"] == "5e838bd6e6db25d223f548bae5bd3419" or \
            doc["id"] == "e142828ec89839a4f500cefed8fb52d3" or \
            doc["id"] == "14a079b9770a338d7da5a996d7a82a83" or \
            doc["id"] == "ee9fd678c60e22593fde13fcc16d71ef" or \
            doc["id"] == "3a1e833be4c09788fe69df8e5c550a27" or \
            doc["id"] == "dbc37e6c17a374526a68e6fb4738b8f7" or \
            doc["id"] == "e7494affe44554efe60498d33040cd5b":
            continue
        result = {"id": doc["id"], "coreference": None, "temporal_relations": {}, "causal_relations": {}, "subevent_relations": None}
        result["coreference"] = []
        result["temporal_relations"] = {"BEFORE": [], "CONTAINS": [], "OVERLAP": [], "BEGINS-ON": [], "ENDS-ON": [], "SIMULTANEOUS": []}
        result["causal_relations"] = {"CAUSE": [], "PRECONDITION": []}
        result["subevent_relations"] = []
        doc["coreference_raw_output"] = []
        doc["temporal_raw_output"] = []
        doc["causal_raw_output"] = []
        doc["subevent_raw_output"] = []
        for relation in ["coreference", "temporal", "causal", "subevent"]:
            if relation == "coreference":
                mentions_appeared = []
                cluster_appeared = []
                coref_tokens = copy.deepcopy(doc["tokens"])
                for j, sentence in enumerate(doc["sentences"]):
                    if j == 0:
                        input_context = sentence
                        mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)
                        for mention in mentions_in_sentence:
                            mentions_appeared.append(mention)
                    else:
                        raw_output = doc[f"{relation}_raw_output"][j-1]
                        raw_output = raw_output.replace(" SHIFT;", "")
                        raw_output = raw_output.split(";")
                        for pair in raw_output:
                            pair = pair.split(" ")
                            if len(pair) != 3:
                                continue
                            in_cluster = False
                            for k, cluster in enumerate(result["coreference"]):
                                if pair[0] in cluster and pair[0] in doc["eventnumberid2mention"]:
                                    result["coreference"][k].append(pair[2])
                                    in_cluster = True
                                elif pair[2] in cluster and pair[2] in doc["eventnumberid2mention"]:
                                    result["coreference"][k].append(pair[0])
                                    in_cluster = True
                            if not in_cluster and pair[0] in doc["eventnumberid2mention"] and pair[2] in doc["eventnumberid2mention"]:
                                result["coreference"].append([pair[0], pair[2]])
                        
                        mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)
                        num = 0
                        for mention in mentions_in_sentence:
                            done = False
                            for mention_appeared in mentions_appeared:
                                for k, cluster in enumerate(result["coreference"]):
                                    if mention in cluster and mention_appeared in cluster:
                                        if k in cluster_appeared:
                                            if done: continue
                                            num += 1
                                            coref_tokens[j].insert(doc["eventnumberid2mention"][mention]["new_offset"][0]+ num, str(k))
                                            done = True
                                            continue
                                        num += 1
                                        coref_tokens[j].insert(doc["eventnumberid2mention"][mention]["new_offset"][0]+num, str(k))
                                        coref_tokens[doc["eventnumberid2mention"][mention_appeared]["sent_id"]].insert(doc["eventnumberid2mention"][mention_appeared]["new_offset"][0] + 1, str(k))
                                        cluster_appeared.append(k)
                            mentions_appeared.append(mention)
                        accumulated_context = [doc["sentences"][index] for index in range(j)]
                        accumulated_context.append(sentence)
                        input_context = ' '.join(join_punctuation(accumulated_context))
                        
                    # Call openai request such as text completion
                    few_shot_message = [{"role": "system", "content": "You are an annotator for the MAVEN-ERE dataset. Your task is to extract event coreference relations between event mentions from given documents, where all event and TIMEX mentions are given in [ ]. Imitate the given example to find coreference relations between event mentions. The last sentence of the context is not annotated. You should find all the relations among the new mentions in the last sentence with mentions in all previous sentences. Predict the relations in this format: Event_1 COREFERENCE Event_0; SHIFT; means moving on to the next sentence. Always add SHIFT; at the end of prediction."}]
                    for q in range(shot):
                        few_shot_message.append({"role": "user", "content": f"{example_docs[q]['coref_context_inputs'][0]}"})
                        few_shot_message.append({"role": "assistant", "content": f"{example_docs[q]['coref_answer_inputs'][0]}"})
                        few_shot_message.append({"role": "user", "content": f"{example_docs[q]['coref_context_inputs'][1]}"})
                        few_shot_message.append({"role": "assistant", "content": f"{example_docs[q]['coref_answer_inputs'][1]}"})
                    few_shot_message.append({"role": "user", "content": f"{input_context}"})

                    dialogs: List[Dialog] = [few_shot_message]

                    results = generator.chat_completion(
                        dialogs,  # type: ignore
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    doc[f"{relation}_raw_output"].append(results[0]['generation']['content'])
                print(f"Doc #{i} coreference result: {result['coreference']}")
                
            elif relation == "temporal":
                mentions_appeared = {}
                mentions_inserted = []
                temporal_tokens = copy.deepcopy(doc["tokens"])
                eventnumberid2appearedrelations = {"BEFORE": {}, "CONTAINS": {}, "OVERLAP": {}, "BEGINS-ON": {}, "ENDS-ON": {}, "SIMULTANEOUS": {}}
                for j, sentence in enumerate(doc["sentences"]):
                    if j == 0:
                        input_context = sentence
                        mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)
                        for mention in mentions_in_sentence:
                            mentions_appeared[mention] = len(mentions_in_sentence)
                    else:
                        raw_output = doc[f"{relation}_raw_output"][j-1]
                        raw_output = raw_output.replace(" SHIFT;", "")
                        raw_output = raw_output.split(";")
                        for pair in raw_output:
                            pair = pair.split(" ")
                            if "" in pair: pair.remove("")
                            if len(pair) != 3:
                                continue
                            if pair[1] in eventnumberid2appearedrelations and pair[0] in doc["eventnumberid2mention"] and pair[2] in doc["eventnumberid2mention"]:
                                result["temporal_relations"][pair[1]].append([pair[0], pair[2]])
                        mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)

                        for mention in mentions_appeared:
                            for mention_appeared in mentions_appeared:
                                for temporal_type in ["BEFORE", "CONTAINS", "OVERLAP", "BEGINS-ON", "ENDS-ON", "SIMULTANEOUS"]:
                                    for pair in result["temporal_relations"][temporal_type]:
                                        if mention == pair[0] and mention_appeared == pair[1]:
                                            if pair[1] in eventnumberid2appearedrelations[temporal_type]:
                                                eventnumberid2appearedrelations[temporal_type][pair[1]].append(f"{mention} {temporal_type} {mention_appeared}")
                                            else:
                                                eventnumberid2appearedrelations[temporal_type][pair[1]] = [f"{mention} {temporal_type} {mention_appeared}"]
                                        elif mention == pair[1] and mention_appeared == pair[0]:
                                            if pair[1] in eventnumberid2appearedrelations[temporal_type]:
                                                eventnumberid2appearedrelations[temporal_type][pair[1]].append(f"{mention_appeared} {temporal_type} {mention}")
                                            else:
                                                eventnumberid2appearedrelations[temporal_type][pair[1]] = [f"{mention_appeared} {temporal_type} {mention}"]

                        for mention in mentions_in_sentence:
                            for mention_appeared in mentions_appeared:
                                for temporal_type in ["BEFORE", "CONTAINS", "OVERLAP", "BEGINS-ON", "ENDS-ON", "SIMULTANEOUS"]:
                                    for pair in result["temporal_relations"][temporal_type]:
                                        if mention == pair[0] and mention_appeared == pair[1]:
                                            if pair[1] in eventnumberid2appearedrelations[temporal_type]:
                                                eventnumberid2appearedrelations[temporal_type][pair[1]].append(f"{mention} {temporal_type} {mention_appeared}")
                                            else:
                                                eventnumberid2appearedrelations[temporal_type][pair[1]] = [f"{mention} {temporal_type} {mention_appeared}"]
                                        elif mention == pair[1] and mention_appeared == pair[0]:
                                            if pair[1] in eventnumberid2appearedrelations[temporal_type]:
                                                eventnumberid2appearedrelations[temporal_type][pair[1]].append(f"{mention_appeared} {temporal_type} {mention}")
                                            else:
                                                eventnumberid2appearedrelations[temporal_type][pair[1]] = [f"{mention_appeared} {temporal_type} {mention}"]
                            mentions_appeared[mention] = len(mentions_in_sentence)

                        num = 0
                        for mention_appeared in mentions_appeared:
                            if mentions_appeared[mention_appeared] == num:
                                num = 0
                            num += 1
                            relations_for_inserting = ""
                            for temporal_type in ["BEFORE", "CONTAINS", "OVERLAP", "BEGINS-ON", "ENDS-ON", "SIMULTANEOUS"]:
                                if mention_appeared in eventnumberid2appearedrelations[temporal_type]:
                                    relations_for_inserting += ';'.join(list(set(eventnumberid2appearedrelations[temporal_type][mention_appeared])))
                                    relations_for_inserting += ';'
                            if mention_appeared in mentions_inserted:
                                temporal_tokens[doc["eventnumberid2mention"][mention_appeared]["sent_id"]][doc["eventnumberid2mention"][mention_appeared]["new_offset"][1] + num] = relations_for_inserting
                            else:
                                temporal_tokens[doc["eventnumberid2mention"][mention_appeared]["sent_id"]].insert(doc["eventnumberid2mention"][mention_appeared]["new_offset"][1] + num, relations_for_inserting)
                                mentions_inserted.append(mention_appeared)
                            
                            if mentions_appeared[mention_appeared] == num:
                                num = 0
                        accumulated_context = [doc["sentences"][index] for index in range(j)]
                        accumulated_context.append(sentence)
                        input_context = ' '.join(join_punctuation(accumulated_context))
                        
                    # Call openai request such as text completion
                    few_shot_message = [{"role": "system", "content": "You are an annotator for the MAVEN-ERE dataset. Your task is to extract temporal relations between event mentions from given documents, where all event and TIMEX mentions are given in [ ]. There are 6 types of temporal relations: BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, and SIMULTANEOUS. Imitate the given example to find temporal relations between event and TIMEX mentions. The last sentence of the context is not annotated. You should find all the relations among the new mentions in the last sentence with mentions in all previous sentences. Predict the relations in this format: Event_1 BEFORE Event_0; SHIFT; means moving on to the next sentence. Always add SHIFT; at the end of prediction."}]
                    for q in range(shot):
                        few_shot_message.append({"role": "user", "content": f"{example_docs[q]['temporal_context_inputs'][0]}"})
                        few_shot_message.append({"role": "assistant", "content": f"{example_docs[q]['temporal_answer_inputs'][0]}"})
                        few_shot_message.append({"role": "user", "content": f"{example_docs[q]['temporal_context_inputs'][1]}"})
                        few_shot_message.append({"role": "assistant", "content": f"{example_docs[q]['temporal_answer_inputs'][1]}"})
                    few_shot_message.append({"role": "user", "content": f"{input_context}"})

                    dialogs: List[Dialog] = [few_shot_message]
                    print(few_shot_message)
                    results = generator.chat_completion(
                        dialogs,  # type: ignore
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    doc[f"{relation}_raw_output"].append(results[0]['generation']['content'])
                print(f"Doc #{i} temporal result: {result['temporal_relations']}")
                
            elif relation == "causal":
                mentions_appeared = {}
                mentions_inserted = []
                causal_tokens = copy.deepcopy(doc["tokens"])
                eventnumberid2appearedrelations = {"CAUSE": {}, "PRECONDITION": {}}
                for j, sentence in enumerate(doc["sentences"]):
                    if j == 0:
                        input_context = sentence
                        mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)
                        for mention in mentions_in_sentence:
                            mentions_appeared[mention] = len(mentions_in_sentence)
                    else:
                        raw_output = doc[f"{relation}_raw_output"][j-1]
                        raw_output = raw_output.replace(" SHIFT;", "")
                        raw_output = raw_output.split(";")
                        for pair in raw_output:
                            pair = pair.split(" ")
                            if "" in pair: pair.remove("")
                            if len(pair) != 3:
                                continue
                            if pair[1] in eventnumberid2appearedrelations and pair[0] in doc["eventnumberid2mention"] and pair[2] in doc["eventnumberid2mention"]:
                                result["causal_relations"][pair[1]].append([pair[0], pair[2]])
                        mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)

                        for mention in mentions_appeared:
                            for mention_appeared in mentions_appeared:
                                for causal_type in ["CAUSE", "PRECONDITION"]:
                                    for pair in result["causal_relations"][causal_type]:
                                        if mention == pair[0] and mention_appeared == pair[1]:
                                            if pair[1] in eventnumberid2appearedrelations[causal_type]:
                                                eventnumberid2appearedrelations[causal_type][pair[1]].append(f"{mention} {causal_type} {mention_appeared}")
                                            else:
                                                eventnumberid2appearedrelations[causal_type][pair[1]] = [f"{mention} {causal_type} {mention_appeared}"]
                                        elif mention == pair[1] and mention_appeared == pair[0]:
                                            if pair[1] in eventnumberid2appearedrelations[causal_type]:
                                                eventnumberid2appearedrelations[causal_type][pair[1]].append(f"{mention_appeared} {causal_type} {mention}")
                                            else:
                                                eventnumberid2appearedrelations[causal_type][pair[1]] = [f"{mention_appeared} {causal_type} {mention}"]

                        for mention in mentions_in_sentence:
                            for mention_appeared in mentions_appeared:
                                for causal_type in ["CAUSE", "PRECONDITION"]:
                                    for pair in result["causal_relations"][causal_type]:
                                        if mention == pair[0] and mention_appeared == pair[1]:
                                            if pair[1] in eventnumberid2appearedrelations[causal_type]:
                                                eventnumberid2appearedrelations[causal_type][pair[1]].append(f"{mention} {causal_type} {mention_appeared}")
                                            else:
                                                eventnumberid2appearedrelations[causal_type][pair[1]] = [f"{mention} {causal_type} {mention_appeared}"]
                                        elif mention == pair[1] and mention_appeared == pair[0]:
                                            if pair[1] in eventnumberid2appearedrelations[causal_type]:
                                                eventnumberid2appearedrelations[causal_type][pair[1]].append(f"{mention_appeared} {causal_type} {mention}")
                                            else:
                                                eventnumberid2appearedrelations[causal_type][pair[1]] = [f"{mention_appeared} {causal_type} {mention}"]
                            mentions_appeared[mention] = len(mentions_in_sentence)

                        num = 0
                        for mention_appeared in mentions_appeared:
                            if mentions_appeared[mention_appeared] == num:
                                num = 0
                            num += 1
                            relations_for_inserting = ""
                            for causal_type in ["CAUSE", "PRECONDITION"]:
                                if mention_appeared in eventnumberid2appearedrelations[causal_type]:
                                    relations_for_inserting += ';'.join(list(set(eventnumberid2appearedrelations[causal_type][mention_appeared])))
                                    relations_for_inserting += ';'
                            if mention_appeared in mentions_inserted:
                                causal_tokens[doc["eventnumberid2mention"][mention_appeared]["sent_id"]][doc["eventnumberid2mention"][mention_appeared]["new_offset"][1] + num] = relations_for_inserting
                            else:
                                causal_tokens[doc["eventnumberid2mention"][mention_appeared]["sent_id"]].insert(doc["eventnumberid2mention"][mention_appeared]["new_offset"][1] + num, relations_for_inserting)
                                mentions_inserted.append(mention_appeared)
                            
                            if mentions_appeared[mention_appeared] == num:
                                num = 0
                        accumulated_context = [doc["sentences"][index] for index in range(j)]
                        accumulated_context.append(sentence)
                        input_context = ' '.join(join_punctuation(accumulated_context))
                        
                    # Call openai request such as text completion
                    few_shot_message = [{"role": "system", "content": "You are an annotator for the MAVEN-ERE dataset. Your task is to extract causal relations between event mentions from given documents, where all event and TIMEX mentions are given in [ ]. There are 2 types of causal relations: CAUSE, and PRECONDITION. Imitate the given example to find causal relations between event and TIMEX mentions. The last sentence of the context is not annotated. You should find all the relations among the new mentions in the last sentence with mentions in all previous sentences. Predict the relations in this format: Event_1 CAUSE Event_0; SHIFT; means moving on to the next sentence. Always add SHIFT; at the end of prediction."}]
                    for q in range(shot):
                        few_shot_message.append({"role": "user", "content": f"{example_docs[q]['causal_context_inputs'][0]}"})
                        few_shot_message.append({"role": "assistant", "content": f"{example_docs[q]['causal_answer_inputs'][0]}"})
                        few_shot_message.append({"role": "user", "content": f"{example_docs[q]['causal_context_inputs'][1]}"})
                        few_shot_message.append({"role": "assistant", "content": f"{example_docs[q]['causal_answer_inputs'][1]}"})
                    few_shot_message.append({"role": "user", "content": f"{input_context}"})

                    dialogs: List[Dialog] = [few_shot_message]

                    results = generator.chat_completion(
                        dialogs,  # type: ignore
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    doc[f"{relation}_raw_output"].append(results[0]['generation']['content'])
                print(f"Doc #{i} causal result: {result['causal_relations']}")
                
            elif relation == "subevent":
                mentions_appeared = {}
                mentions_inserted = []
                subevent_tokens = copy.deepcopy(doc["tokens"])
                eventnumberid2appearedrelations = {"SUBEVENT": {}}
                for j, sentence in enumerate(doc["sentences"]):
                    if j >= len(example_doc["sentences"]) - 1:
                        prompt_context = example_doc["subevent_context_inputs"][-1]
                        prompt_answer = example_doc["subevent_answer_inputs"][-1]
                    else:
                        prompt_context = example_doc["subevent_context_inputs"][j]
                        prompt_answer = example_doc["subevent_answer_inputs"][j]

                    if j == 0:
                        input_context = sentence
                        mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)
                        for mention in mentions_in_sentence:
                            mentions_appeared[mention] = len(mentions_in_sentence)
                    else:
                        raw_output = doc[f"{relation}_raw_output"][j-1]
                        raw_output = raw_output.replace(" SHIFT;", "")
                        raw_output = raw_output.split(";")
                        for pair in raw_output:
                            pair = pair.split(" ")
                            if "" in pair: pair.remove("")
                            if len(pair) != 3:
                                continue
                            if pair[1] in eventnumberid2appearedrelations and pair[0] in doc["eventnumberid2mention"] and pair[2] in doc["eventnumberid2mention"]:
                                result["subevent_relations"].append([pair[0], pair[2]])
                        mentions_in_sentence = re.findall(r'TIMEX_[0-9]*|Event_[0-9]*', sentence)

                        for mention in mentions_appeared:
                            for mention_appeared in mentions_appeared:
                                for subevent_type in ["SUBEVENT"]:
                                    for pair in result["subevent_relations"]:
                                        if mention == pair[0] and mention_appeared == pair[1]:
                                            if pair[1] in eventnumberid2appearedrelations[subevent_type]:
                                                eventnumberid2appearedrelations[subevent_type][pair[1]].append(f"{mention} {subevent_type} {mention_appeared}")
                                            else:
                                                eventnumberid2appearedrelations[subevent_type][pair[1]] = [f"{mention} {subevent_type} {mention_appeared}"]
                                        elif mention == pair[1] and mention_appeared == pair[0]:
                                            if pair[1] in eventnumberid2appearedrelations[subevent_type]:
                                                eventnumberid2appearedrelations[subevent_type][pair[1]].append(f"{mention_appeared} {subevent_type} {mention}")
                                            else:
                                                eventnumberid2appearedrelations[subevent_type][pair[1]] = [f"{mention_appeared} {subevent_type} {mention}"]

                        for mention in mentions_in_sentence:
                            for mention_appeared in mentions_appeared:
                                for subevent_type in ["SUBEVENT"]:
                                    for pair in result["subevent_relations"]:
                                        if mention == pair[0] and mention_appeared == pair[1]:
                                            if pair[1] in eventnumberid2appearedrelations[subevent_type]:
                                                eventnumberid2appearedrelations[subevent_type][pair[1]].append(f"{mention} {subevent_type} {mention_appeared}")
                                            else:
                                                eventnumberid2appearedrelations[subevent_type][pair[1]] = [f"{mention} {subevent_type} {mention_appeared}"]
                                        elif mention == pair[1] and mention_appeared == pair[0]:
                                            if pair[1] in eventnumberid2appearedrelations[subevent_type]:
                                                eventnumberid2appearedrelations[subevent_type][pair[1]].append(f"{mention_appeared} {subevent_type} {mention}")
                                            else:
                                                eventnumberid2appearedrelations[subevent_type][pair[1]] = [f"{mention_appeared} {subevent_type} {mention}"]
                            mentions_appeared[mention] = len(mentions_in_sentence)

                        num = 0
                        for mention_appeared in mentions_appeared:
                            if mentions_appeared[mention_appeared] == num:
                                num = 0
                            num += 1
                            relations_for_inserting = ""
                            for subevent_type in ["SUBEVENT"]:
                                if mention_appeared in eventnumberid2appearedrelations[subevent_type]:
                                    relations_for_inserting += ';'.join(list(set(eventnumberid2appearedrelations[subevent_type][mention_appeared])))
                                    relations_for_inserting += ';'
                            if mention_appeared in mentions_inserted:
                                subevent_tokens[doc["eventnumberid2mention"][mention_appeared]["sent_id"]][doc["eventnumberid2mention"][mention_appeared]["new_offset"][1] + num] = relations_for_inserting
                            else:
                                subevent_tokens[doc["eventnumberid2mention"][mention_appeared]["sent_id"]].insert(doc["eventnumberid2mention"][mention_appeared]["new_offset"][1] + num, relations_for_inserting)
                                mentions_inserted.append(mention_appeared)
                            
                            if mentions_appeared[mention_appeared] == num:
                                num = 0
                        accumulated_context = [doc["sentences"][index] for index in range(j)]
                        accumulated_context.append(sentence)
                        input_context = ' '.join(join_punctuation(accumulated_context))
                        
                    # Call openai request such as text completion
                    few_shot_message = [{"role": "system", "content": "You are an annotator for the MAVEN-ERE dataset. Your task is to extract subevent relations between event mentions from given documents, where all event and TIMEX mentions are given in [ ]. Imitate the given example to find subevent relations between event and TIMEX mentions. The last sentence of the context is not annotated. You should find all the relations among the new mentions in the last sentence with mentions in all previous sentences. Predict the relations in this format: Event_1 SUBEVENT Event_0; SHIFT; means moving on to the next sentence. Always add SHIFT; at the end of prediction."}]
                    for q in range(shot):
                        few_shot_message.append({"role": "user", "content": f"{example_docs[q]['subevent_context_inputs'][0]}"})
                        few_shot_message.append({"role": "assistant", "content": f"{example_docs[q]['subevent_answer_inputs'][0]}"})
                        few_shot_message.append({"role": "user", "content": f"{example_docs[q]['subevent_context_inputs'][1]}"})
                        few_shot_message.append({"role": "assistant", "content": f"{example_docs[q]['subevent_answer_inputs'][1]}"})
                    few_shot_message.append({"role": "user", "content": f"{input_context}"})

                    dialogs: List[Dialog] = [few_shot_message]

                    results = generator.chat_completion(
                        dialogs,  # type: ignore
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    doc[f"{relation}_raw_output"].append(results[0]['generation']['content'])
                print(f"Doc #{i} subevent result: {result['subevent_relations']}")
        
        if not test:
            with open(os.path.join(output_dir, f"test_prediction.jsonl"), "a")as f:
                f.write(json.dumps(result))
                f.write("\n")
        else:
            with open(os.path.join(output_dir, f"test_prediction_{partition}.jsonl"), "a")as f:
                f.write(json.dumps(result))
                f.write("\n")

if __name__ == "__main__":
    fire.Fire(main)