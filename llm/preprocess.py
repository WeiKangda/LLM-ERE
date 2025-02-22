from tqdm import tqdm
import json
from collections import defaultdict
import re
import argparse
import random
import time
import copy
import os
import random

def process_prediction(pred, relation):

    if relation == "coreference" or relation == "subevent":
        if "Yes" in pred:
            result = "Yes"
        else:
            result = "No"
    elif relation == "temporal":
        if "BEFORE" in pred:
            result = "BEFORE"
        elif "CONTAINS" in pred:
            result = "CONTAINS"
        elif "OVERLAP" in pred:
            result = "OVERLAP"
        elif "BEGINS-ON" in pred:
            result = "BEGINS-ON"
        elif "ENDS-ON" in pred:
            result = "ENDS-ON"
        elif "SIMULTANEOUS" in pred:
            result = "SIMULTANEOUS"
        else:
            result = "NONE"
    elif relation == "causal":
        if "CAUSE" in pred:
            result = "CAUSE"
        elif "PRECONDITION" in pred:
            result = "PRECONDITION"
        else:
            result = "NONE"
    
    return result
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

def valid_first_10_doc_process(data_dir):
    docs = []
    all_eventnumberid2mentionid = {}
    with open(os.path.join(data_dir, f"valid_first10.jsonl"))as f:
            lines = f.readlines()
            for i, line in enumerate(tqdm(lines, desc="Loading test data")):
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
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"<EVENT>")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"</EVENT>")
                for mention in doc["TIMEX"]:
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"<TIMEX>")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"</TIMEX>")
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
                    doc["subevent_answer_inputs"].append(subevent_answer_input)
                    doc["subevent_context_inputs"].append(accumulated_context)
                all_eventnumberid2mentionid[doc["id"]] = eventnumberid2mentionid
                docs.append(doc)
    return docs

def test_doc_process(data_dir):
    docs = []
    all_eventnumberid2mentionid = {}
    with open(os.path.join(data_dir, f"test.jsonl"))as f:
            lines = f.readlines()
            for i, line in enumerate(tqdm(lines, desc="Loading test data")):
                doc = json.loads(line.strip())
                eventnumberid2mentionid = {}
                mention_in_sent = defaultdict(list)
                mentionid2eventnumberid = {}

                eventnumberid2mention = {}

                doc["event_mentions"] = sorted(doc["event_mentions"], key=lambda x: (x["sent_id"], x["offset"][0]))
                
                id_in_event_number = 0
                '''for event in doc["events"]:
                    if len(event["mention"]) != 1:
                            for i in range(len(event["mention"]) - 1):
                                doc["corefernce_relations"].append([event["mention"][i]["id"], event["mention"][i+1]["id"]])

                    for mention in event["mention"]:
                        eventid2mentionid[event["id"]].append(mention["id"])'''
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
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"<EVENT>")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"</EVENT>")
                for mention in doc["TIMEX"]:
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"<TIMEX>")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"</TIMEX>")
                doc["sentences"] = [' '.join(join_punctuation(sentence)) for sentence in doc["tokens"]]
                doc["context"] = ' '.join(doc["sentences"])
                #print(doc)
                all_eventnumberid2mentionid[doc["id"]] = eventnumberid2mentionid
                docs.append(doc)
    return docs

def example_docs_process():
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
                example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"<EVENT>")
                example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"</EVENT>")
            for mention in example_doc["TIMEX"]:
                example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"<TIMEX>")
                example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"</TIMEX>")
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
                example_doc["subevent_answer_inputs"].append(subevent_answer_input)
                example_doc["subevent_context_inputs"].append(accumulated_context)
            example_docs.append(example_doc)
    return example_docs

def example_doc_process():
    with open(f"./example.jsonl")as f:
        lines = f.readlines()
        example_doc = json.loads(lines[0].strip())
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
            example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"<EVENT>")
            example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
            example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"</EVENT>")
        for mention in example_doc["TIMEX"]:
            example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"<TIMEX>")
            example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
            example_doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"</TIMEX>")
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
    return example_doc

def train_or_valid_doc_process(data_dir, split="train"):
    docs = []
    all_eventnumberid2mentionid = {}
    if split == "train":
        file_name = "train.jsonl"
    else:
        file_name = "valid.jsonl"
    with open(os.path.join(data_dir, file_name))as f:
            lines = f.readlines()
            for i, line in enumerate(tqdm(lines, desc=f"Loading {split} data")):
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
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"<EVENT>")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"</EVENT>")
                for mention in doc["TIMEX"]:
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][0], f"<TIMEX>")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1], f"{mention['id_in_number']}")
                    doc["tokens"][mention["sent_id"]].insert(mention["new_offset"][1]+1, f"</TIMEX>")
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
                    doc["subevent_answer_inputs"].append(subevent_answer_input)
                    doc["subevent_context_inputs"].append(accumulated_context)
                all_eventnumberid2mentionid[doc["id"]] = eventnumberid2mentionid
                docs.append(doc)
    return docs

def check_has_pair(p, pairs):
    for pair in pairs:
        if pair[0] == p[0] and pair[1] == p[1]:
            return True
    return False

def docs_to_pairs(docs, split="train", relation="coreference", down_sample="with_downsample",doc_number=10, coreference_none_drop_rate=0.95, temporal_none_drop_rate=0.90, causal_none_drop_rate=0.98, subevent_none_drop_rate=0.99, before_drop_rate=0.90):
    data_dir = "../data/MAVEN_ERE"
    context_dict = {}
    for doc in tqdm(docs, desc=f"Converting {split} data from document to pairwise"):
        context_dict[doc["id"]] = doc["context"]
        if relation == "coreference" or relation == "joint":
            for mention_i in doc['event_mentions']:
                event_i = mention_i['id_in_number']
                mention_i = '<EVENT> ' + mention_i['trigger_word'] + ' ' + mention_i['id_in_number'] + '</EVENT>'
                for mention_j in doc['event_mentions']:
                    event_j = mention_j['id_in_number']
                    mention_j = '<EVENT> ' + mention_j['trigger_word'] + ' ' + mention_j['id_in_number'] + '</EVENT>'
                    if mention_i != mention_j:
                        #for relation in ["coreference", "temporal", "causal", "subevent"]:
                        #if relation == "coreference" or relation == "joint":
                        if split == "test" or split == "valid":
                            processed_data_point = {"prompt": f" What is the coreference relation between {mention_i} and {mention_j}? Answer from [NONE, COREFERENCE]", "completion": "", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                        else:
                            if check_has_pair([event_i, event_j], doc["coreference_relations"]):
                                processed_data_point = {"prompt": f" What is the coreference relation between {mention_i} and {mention_j}? Answer from [NONE, COREFERENCE]", "completion": "COREFERENCE", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                            else:
                                if random.uniform(0, 1) > coreference_none_drop_rate:
                                    processed_data_point = {"prompt": f" What is the coreference relation between {mention_i} and {mention_j}? Answer from [NONE, COREFERENCE]", "completion": "NONE", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                                else:
                                    continue
                        with open(os.path.join(data_dir, f"{split}_{relation}_{doc_number}_processed_{down_sample}.jsonl"), "a")as f:
                            f.write(json.dumps(processed_data_point))
                            f.write("\n")
        if relation == "causal" or relation == "joint":
            for mention_i in doc['event_mentions']:
                event_i = mention_i['id_in_number']
                mention_i = '<EVENT> ' + mention_i['trigger_word'] + ' ' + mention_i['id_in_number'] + '</EVENT>'
                for mention_j in doc['event_mentions']:
                    event_j = mention_j['id_in_number']
                    mention_j = '<EVENT> ' + mention_j['trigger_word'] + ' ' + mention_j['id_in_number'] + '</EVENT>'
                    if mention_i != mention_j:
                        if split == "test" or split == "valid":
                                processed_data_point = {"prompt": f" What is the causal relation between {mention_i} and {mention_j}? Answer from [NONE, CAUSE, PRECONDITION]", "completion": "", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                        else:
                            if check_has_pair([event_i, event_j], doc["causal_relations"]["CAUSE"]):
                                processed_data_point = {"prompt": f" What is the causal relation between {mention_i} and {mention_j}? Answer from [NONE, CAUSE, PRECONDITION]", "completion": "CAUSE", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                            elif check_has_pair([event_i, event_j], doc["causal_relations"]["PRECONDITION"]):
                                processed_data_point = {"prompt": f" What is the causal relation between {mention_i} and {mention_j}? Answer from [NONE, CAUSE, PRECONDITION]", "completion": "PRECONDITION", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                            else:
                                if random.uniform(0, 1) > causal_none_drop_rate:
                                    processed_data_point = {"prompt":f" What is the causal relation between {mention_i} and {mention_j}? Answer from [NONE, CAUSE, PRECONDITION]", "completion": "NONE", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                                else:
                                    continue
                        with open(os.path.join(data_dir, f"{split}_{relation}_{doc_number}_processed_{down_sample}.jsonl"), "a")as f:
                            f.write(json.dumps(processed_data_point))
                            f.write("\n")
        if relation == "subevent" or relation == "joint":
            for mention_i in doc['event_mentions']:
                event_i = mention_i['id_in_number']
                mention_i = '<EVENT> ' + mention_i['trigger_word'] + ' ' + mention_i['id_in_number'] + '</EVENT>'
                for mention_j in doc['event_mentions']:
                    event_j = mention_j['id_in_number']
                    mention_j = '<EVENT> ' + mention_j['trigger_word'] + ' ' + mention_j['id_in_number'] + '</EVENT>'
                    if mention_i != mention_j:
                        if split == "test" or split == "valid":
                                processed_data_point = {"prompt": f" What is the subevent relation between {mention_i} and {mention_j}? Answer from [NONE, SUBEVENT]", "completion": "", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                        else:
                            if check_has_pair([event_i, event_j], doc["subevent_relations"]):
                                processed_data_point = {"prompt": f" What is the subevent relation between {mention_i} and {mention_j}? Answer from [NONE, SUBEVENT]", "completion": "SUBEVENT", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                            else:
                                if random.uniform(0, 1) > subevent_none_drop_rate:
                                    processed_data_point = {"prompt": f" What is the subevent relation between {mention_i} and {mention_j}? Answer from [NONE, SUBEVENT]", "completion": "NONE", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                                else:
                                    continue
                        with open(os.path.join(data_dir, f"{split}_{relation}_{doc_number}_processed_{down_sample}.jsonl"), "a")as f:
                            f.write(json.dumps(processed_data_point))
                            f.write("\n")
                            
        if relation == "temporal" or relation == "joint":
            for mention_i in doc['all_mentions']:
                event_i = mention_i['id_in_number']
                #print(mention_i)
                if mention_i['id_in_number'].startswith('Event'):
                    mention_i = '<EVENT> ' + mention_i['trigger_word'] + ' ' + mention_i['id_in_number'] + '</EVENT>'
                else:
                    mention_i = '<TIMEX> ' + mention_i['mention'] + ' ' + mention_i['id_in_number'] + '</TIMEX>'
                for mention_j in doc['all_mentions']:
                    event_j = mention_j['id_in_number']
                    if mention_j['id_in_number'].startswith('Event'):
                        mention_j = '<EVENT> ' + mention_j['trigger_word'] + ' ' + mention_j['id_in_number'] + '</EVENT>'
                    else:
                        mention_j = '<TIMEX> ' + mention_j['mention'] + ' ' + mention_j['id_in_number'] + '</TIMEX>'
                    if mention_i != mention_j:
                        if split == "test" or split == "valid":
                            processed_data_point = {"prompt": f" What is the temporal relation between {mention_i} and {mention_j}? Answer from [NONE, BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS]", "completion": "", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                        else:
                            if check_has_pair([event_i, event_j], doc["temporal_relations"]["BEFORE"]):
                                if random.uniform(0, 1) > before_drop_rate:
                                    processed_data_point = {"prompt": f" What is the temporal relation between {mention_i} and {mention_j}? Answer from [NONE, BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS]", "completion": "BEFORE", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                                else:
                                    continue
                            elif check_has_pair([event_i, event_j], doc["temporal_relations"]["CONTAINS"]):
                                processed_data_point = {"prompt": f" What is the temporal relation between {mention_i} and {mention_j}? Answer from [NONE, BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS]", "completion": "CONTAINS", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                            elif check_has_pair([event_i, event_j], doc["temporal_relations"]["OVERLAP"]):
                                processed_data_point = {"prompt": f" What is the temporal relation between {mention_i} and {mention_j}? Answer from [NONE, BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS]", "completion": "OVERLAP", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                            elif check_has_pair([event_i, event_j], doc["temporal_relations"]["BEGINS-ON"]):
                                processed_data_point = {"prompt": f" What is the temporal relation between {mention_i} and {mention_j}? Answer from [NONE, BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS]", "completion": "BEGINS-ON", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                            elif check_has_pair([event_i, event_j], doc["temporal_relations"]["ENDS-ON"]):
                                processed_data_point = {"prompt": f" What is the temporal relation between {mention_i} and {mention_j}? Answer from [NONE, BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS]", "completion": "ENDS-ON", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                            elif check_has_pair([event_i, event_j], doc["temporal_relations"]["SIMULTANEOUS"]):
                                processed_data_point = {"prompt": f" What is the temporal relation between {mention_i} and {mention_j}? Answer from [NONE, BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS]", "completion": "SIMULTANEOUS", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                            else:
                                if random.uniform(0, 1) > temporal_none_drop_rate:
                                    processed_data_point = {"prompt": f" What is the temporal relation between {mention_i} and {mention_j}? Answer from [NONE, BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, SIMULTANEOUS]", "completion": "NONE", "doc_id": doc["id"], "event_1": event_i, "event_2": event_j}
                                else:
                                    continue
                        with open(os.path.join(data_dir, f"{split}_{relation}_{doc_number}_processed_{down_sample}.jsonl"), "a")as f:
                            f.write(json.dumps(processed_data_point))
                            f.write("\n")
    if not os.path.exists(os.path.join(data_dir, f"{split}_{relation}_{doc_number}_context_dict_{down_sample}.jsonl")):
        with open(os.path.join(data_dir, f"{split}_{relation}_{doc_number}_context_dict_{down_sample}.jsonl"), "a")as f:
            f.write(json.dumps(context_dict))
            f.write("\n")

def transform_train_conversation(example, context_dict):
    prompt= example['prompt']
    answer = example['completion']
    return {'text': f'<s>[INST] <<SYS>> Predict the relation between different EVENT and TIMEX enclosed in <EVENT></EVENT> and <TIMEX></TIMEX> <</SYS>> Given the document D: {context_dict[example["doc_id"]] + prompt} [/INST] {answer} </s>', "doc_id": example["doc_id"], "event_1": example["event_1"], "event_2": example["event_2"]}

def transform_valid_test_conversation(example, context_dict):
    prompt= example['prompt']
    answer = example['completion']
    return {'text': f'<s>[INST] <<SYS>> Predict the relation between different EVENT and TIMEX enclosed in <EVENT></EVENT> and <TIMEX></TIMEX> respectively <</SYS>> Given the document D: {context_dict[example["doc_id"]] + prompt} [/INST]', "doc_id": example["doc_id"], "event_1": example["event_1"], "event_2": example["event_2"]}

def open_context_file(path):
    with open(path, "r")as f:
        lines = f.readlines()
        context_file = json.loads(lines[0].strip())
    return context_file