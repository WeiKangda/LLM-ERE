import os
import openai
#from .data import *
from tqdm import tqdm
import sys
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_eval", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    openai.api_key = os.getenv("OPENAI_API_KEY")

    docs = []
    if args.test: 
        output_dir = "./llm/output/gpt4_event_ranking/test"
    else:
        output_dir = "./llm/output/gpt4_event_ranking/valid"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dir = "./data/MAVEN_ERE"

    if os.path.exists(os.path.join(output_dir, "log.txt")):
        os.remove(os.path.join(output_dir, "log.txt"))
    sys.stdout = open(os.path.join(output_dir, "log.txt"), 'w')

    if os.path.exists(os.path.join(output_dir, "test_prediction.jsonl")) and not args.resume_eval:
        os.remove(os.path.join(output_dir, "test_prediction.jsonl"))

    #with open(os.path.join(data_dir, f"test.jsonl"))as f:
    #    lines = f.readlines()
    #    for line in tqdm(lines, desc="Loading test data"):
    #        doc = json.loads(line.strip())
    #        #doc = Document(data, ignore_nonetype=False)
    #        docs.append(doc)

    resume_doc_number = 0
    if args.resume_eval:
        with open(os.path.join(output_dir, f"test_prediction.jsonl"), "r")as f:
            lines = f.readlines()
            for line in lines:
                resume_doc_number += 1

    all_eventnumberid2mentionid = {}
    if not args.test:
        with open(os.path.join(data_dir, f"valid_first10.jsonl"))as f:
            lines = f.readlines()
            for i, line in enumerate(tqdm(lines, desc="Loading test data")):

                if args.resume_eval and i < resume_doc_number:
                    continue
                doc = json.loads(line.strip())
                eventnumberid2mentionid = {}
                doc["event_mentions"] = []
                mention_in_sent = defaultdict(list)
                doc["corefernce_relations"] = []
                eventid2mentionid = defaultdict(list)
                mentionid2eventnumberid = {}

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
                            num += 1
                    mention["offset"] = [mention["offset"][0] + num, mention["offset"][1] + num]
                    mention["id_in_number"] = f"Event_{id_in_event_number}"
                    mention_in_sent[mention["sent_id"]].append(mention)
                    mentionid2eventnumberid[mention["id"]] = mention["id_in_number"]
                    eventnumberid2mentionid[mention["id_in_number"]] = mention["id"]
                    id_in_event_number += 1
                
                doc["TIMEX"] = sorted(doc["TIMEX"], key=lambda x: (x["sent_id"], x["offset"][0]))

                id_in_timex_number = 0 
                for mention in doc["TIMEX"]:
                    num = 0
                    for previous_mention in mention_in_sent[mention["sent_id"]]:
                        if previous_mention["offset"][1] < mention["offset"][1]:
                            num += 1
                    mention["offset"] = [mention["offset"][0] + num, mention["offset"][1] + num]
                    mention["id_in_number"] = f"TIMEX_{id_in_timex_number}"
                    mention_in_sent[mention["sent_id"]].append(mention)
                    eventid2mentionid[mention["id"]].append(mention["id"])
                    mentionid2eventnumberid[mention["id"]] = mention["id_in_number"]
                    eventnumberid2mentionid[mention["id_in_number"]] = mention["id"]
                    id_in_timex_number += 1 

                doc["all_mentions"] = copy.deepcopy(doc["event_mentions"])
                doc["all_mentions"].extend(copy.deepcopy(doc["TIMEX"]))
                #print(doc["all_mentions"])
                    
                doc["context"] = doc["tokens"]
                for mention in doc["event_mentions"]:
                    doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id_in_number']}]")
                for mention in doc["TIMEX"]:
                    doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id_in_number']}]")
                doc["context"] = [' '.join(join_punctuation(sentence)) for sentence in doc["context"]]
                doc["context"] = ' '.join(doc["context"])

                mentions_in_order = re.findall(r'\[.*?\]', doc["context"])
                #print(str(mentions_in_order))
                for i, mention in enumerate(mentions_in_order):
                    mentions_in_order[i] = mention[1:-1]
                #print(str(mentions_in_order))
                mentions_in_order_dict = {}
                for i, mention in enumerate(mentions_in_order):
                    mentions_in_order_dict[mention] = i
                #print(mentions_in_order_dict)

                #print(mentionid2eventnumberid)
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
                #print(coreference_relations_event_order)
                #print(f"{doc['id']} : {doc['context']}")
                #print(f"coreference_relations: {doc['coreference_relations']}, temporal_relations: {doc['temporal_relations']}, causal_relations: {doc['causal_relations']}, subevent_relations: {doc['subevent_relations']}")
                all_eventnumberid2mentionid[doc["id"]] = eventnumberid2mentionid

                docs.append(doc)
    else:
        with open(os.path.join(data_dir, f"test.jsonl"))as f:
            lines = f.readlines()
            for i, line in enumerate(tqdm(lines, desc="Loading test data")):

                if args.resume_eval and i < resume_doc_number:
                    continue
                doc = json.loads(line.strip())
                eventnumberid2mentionid = {}
                doc["event_mentions"] = []
                mention_in_sent = defaultdict(list)
                doc["corefernce_relations"] = []
                eventid2mentionid = defaultdict(list)
                mentionid2eventnumberid = {}

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
                            num += 1
                    mention["offset"] = [mention["offset"][0] + num, mention["offset"][1] + num]
                    mention["id_in_event_number"] = f"Event_{id_in_event_number}"
                    mention_in_sent[mention["sent_id"]].append(mention)
                    mentionid2eventnumberid[mention["id"]] = mention["id_in_event_number"]
                    eventnumberid2mentionid[mention["id_in_event_number"]] = mention["id"]
                    id_in_event_number += 1
                
                doc["TIMEX"] = sorted(doc["TIMEX"], key=lambda x: (x["sent_id"], x["offset"][0]))

                id_in_timex_number = 0 
                for mention in doc["TIMEX"]:
                    num = 0
                    for previous_mention in mention_in_sent[mention["sent_id"]]:
                        if previous_mention["offset"][1] < mention["offset"][1]:
                            num += 1
                    mention["offset"] = [mention["offset"][0] + num, mention["offset"][1] + num]
                    mention["id_in_timex_number"] = f"TIMEX_{id_in_timex_number}"
                    mention_in_sent[mention["sent_id"]].append(mention)
                    eventid2mentionid[mention["id"]].append(mention["id"])
                    mentionid2eventnumberid[mention["id"]] = mention["id_in_timex_number"]
                    eventnumberid2mentionid[mention["id_in_timex_number"]] = mention["id"]
                    id_in_timex_number += 1 
                    
                doc["context"] = doc["tokens"]
                for mention in doc["event_mentions"]:
                    doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id_in_event_number']}]")
                for mention in doc["TIMEX"]:
                    doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id_in_timex_number']}]")
                doc["context"] = [' '.join(join_punctuation(sentence)) for sentence in doc["context"]]
                doc["context"] = ' '.join(doc["context"])

                mentions_in_order = re.findall(r'\[.*?\]', doc["context"])
                #print(str(mentions_in_order))
                for i, mention in enumerate(mentions_in_order):
                    mentions_in_order[i] = mention[1:-1]
                #print(str(mentions_in_order))
                mentions_in_order_dict = {}
                for i, mention in enumerate(mentions_in_order):
                    mentions_in_order_dict[mention] = i
                #print(mentions_in_order_dict)

                #print(mentionid2eventnumberid)
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

                #print(coreference_relations_event_order)
                #print(f"{doc['id']} : {doc['context']}")
                #print(f"coreference_relations: {doc['coreference_relations']}, temporal_relations: {doc['temporal_relations']}, causal_relations: {doc['causal_relations']}, subevent_relations: {doc['subevent_relations']}")
                all_eventnumberid2mentionid[doc["id"]] = eventnumberid2mentionid

                docs.append(doc)

    if not args.resume_eval:
        with open(os.path.join(output_dir, "eventnumberid2mentionid.jsonl"), "w") as outfile:
            json.dump(all_eventnumberid2mentionid, outfile)

    example_doc = None
    with open(f"./llm/example.jsonl")as f:
        lines = f.readlines()
        example_doc = json.loads(lines[0].strip())
        example_doc["event_mentions"] = []
        mention_in_sent = defaultdict(list)
        example_doc["corefernce_relations"] = []
        eventid2mentionid = defaultdict(list)
        mentionid2eventnumberid = {}

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
                    num += 1
            mention["offset"] = [mention["offset"][0] + num, mention["offset"][1] + num]
            mention["id_in_number"] = f"Event_{id_in_event_number}"
            mention_in_sent[mention["sent_id"]].append(mention)
            mentionid2eventnumberid[mention["id"]] = mention["id_in_number"]
            id_in_event_number += 1
        
        example_doc["TIMEX"] = sorted(example_doc["TIMEX"], key=lambda x: (x["sent_id"], x["offset"][0]))

        id_in_timex_number = 0 
        for mention in example_doc["TIMEX"]:
            num = 0
            for previous_mention in mention_in_sent[mention["sent_id"]]:
                if previous_mention["offset"][1] < mention["offset"][1]:
                    num += 1
            mention["offset"] = [mention["offset"][0] + num, mention["offset"][1] + num]
            mention["id_in_number"] = f"TIMEX_{id_in_timex_number}"
            mention_in_sent[mention["sent_id"]].append(mention)
            eventid2mentionid[mention["id"]].append(mention["id"])
            mentionid2eventnumberid[mention["id"]] = mention["id_in_number"]
            id_in_timex_number += 1 

        example_doc["all_mentions"] = copy.deepcopy(example_doc["event_mentions"])
        example_doc["all_mentions"].extend(copy.deepcopy(example_doc["TIMEX"]))
            
        example_doc["context"] = example_doc["tokens"]
        for mention in example_doc["event_mentions"]:
            example_doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id_in_number']}]")
        for mention in example_doc["TIMEX"]:
            example_doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id_in_number']}]")
        example_doc["context"] = [' '.join(join_punctuation(sentence)) for sentence in example_doc["context"]]
        example_doc["context"] = ' '.join(example_doc["context"])

        mentions_in_order = re.findall(r'\[.*?\]', example_doc["context"])
        #print(str(mentions_in_order))
        for i, mention in enumerate(mentions_in_order):
            mentions_in_order[i] = mention[1:-1]
        #print(str(mentions_in_order))
        mentions_in_order_dict = {}
        for i, mention in enumerate(mentions_in_order):
            mentions_in_order_dict[mention] = i
        #print(mentions_in_order_dict)

        #print(mentionid2eventnumberid)
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
        print(example_doc)
        #print(example_doc["context"])

    for i, doc in enumerate(tqdm(docs, desc="Predicting")):
        #if i == 1:
        #    exit()
        result = {"id": doc["id"], "coreference": None, "temporal_relations": {}, "causal_relations": {}, "subevent_relations": None}
        result["coreference"] = {}
        result["temporal_relations"] = {"BEFORE": {}, "CONTAINS": {}, "OVERLAP": {}, "BEGINS-ON": {}, "ENDS-ON": {}, "SIMULTANEOUS": {}}
        result["causal_relations"] = {"CAUSE": {}, "PRECONDITION": {}}
        result["subevent_relations"] = {}
        for relation in ["coreference", "temporal", "causal", "subevent"]:
            if relation == "coreference":
                for mention in doc["event_mentions"]:
                    for delay_secs in (2**x for x in range(0, 6)):
                        try:
                        # Call openai request such as text completion
                            if example_doc['event_mentions'][0]['id_in_number'] in example_doc['coreference_relations_event_order']:
                                example_answer = example_doc['coreference_relations_event_order'][example_doc['event_mentions'][0]['id_in_number']]
                            else:
                                example_answer = []
                            completion = openai.ChatCompletion.create(
                            model="gpt-4-1106-preview",
                            messages=[
                                {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given in [] after triggering words. All predictions should be an array with elements being EVENT and TIMEX mentions given in [] from document."},
                                {"role": "user", "content": f"This is the document: {example_doc['context']}."},
                                {"role": "user", "content": f"List all mentions that are COREFERENENCE of {example_doc['event_mentions'][0]['id_in_number']}. If there is no relations, return an empty array."},
                                {"role": "assistant", "content": f"{example_answer}"},
                                {"role": "user", "content": f"This is the document: {doc['context']}."},
                                {"role": "user", "content": f"List all mentions that are COREFERENENCE of {mention['id_in_number']}. If there is no relations, return an empty array."},
                            ])
                                
                            result["coreference"][mention["id_in_number"]] = completion.choices[0].message["content"]
                            break
        
                        except openai.OpenAIError as e:
                            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                            sleep_dur = delay_secs + randomness_collision_avoidance
                            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                            time.sleep(sleep_dur)
                            continue
                print(f"Doc #{i} coreference result: {result['coreference']}")
            elif relation == "temporal":
                for sub_relation in ["BEFORE", "CONTAINS", "OVERLAP", "BEGINS-ON", "ENDS-ON", "SIMULTANEOUS"]:
                    if sub_relation == "BEFORE":
                        for mention in doc["all_mentions"]:
                            for delay_secs in (2**x for x in range(0, 6)):
                                try:
                                # Call openai request such as text completion
                                    if example_doc['event_mentions'][0]['id_in_number'] in example_doc['temporal_relations_event_order'][sub_relation]:
                                        example_answer = example_doc['temporal_relations_event_order'][sub_relation][example_doc['event_mentions'][0]['id_in_number']]
                                    else:
                                        example_answer = []
                                    completion = openai.ChatCompletion.create(
                                    model="gpt-4-1106-preview",
                                    messages=[
                                        {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given in [] after triggering words. All predictions should be an array with elements being EVENT and TIMEX mentions given in [] from document."},
                                        {"role": "user", "content": f"This is the document: {example_doc['context']}."},
                                        {"role": "user", "content": f"List all mentions happened BEFORE {example_doc['event_mentions'][0]['id_in_number']}. If there is no relations, return an empty array."},
                                        {"role": "assistant", "content": f"{example_answer}"},
                                        {"role": "user", "content": f"This is the document: {doc['context']}."},
                                        {"role": "user", "content": f"List all mentions happened BEFORE {mention['id_in_number']}. If there is no relations, return an empty array."},
                                    ])

                                    result["temporal_relations"][sub_relation][mention["id_in_number"]] = completion.choices[0].message["content"]
                                    break
                
                                except openai.OpenAIError as e:
                                    randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                                    sleep_dur = delay_secs + randomness_collision_avoidance
                                    print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                                    time.sleep(sleep_dur)
                                    continue
                
                    if sub_relation == "CONTAINS":
                        for mention in doc["all_mentions"]:
                            for delay_secs in (2**x for x in range(0, 6)):
                                try:
                                # Call openai request such as text completion
                                    if example_doc['event_mentions'][0]['id_in_number'] in example_doc['temporal_relations_event_order'][sub_relation]:
                                        example_answer = example_doc['temporal_relations_event_order'][sub_relation][example_doc['event_mentions'][0]['id_in_number']]
                                    else:
                                        example_answer = []
                                    completion = openai.ChatCompletion.create(
                                    model="gpt-4-1106-preview",
                                    messages=[
                                        {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given in [] after triggering words. All predictions should be an array with elements being EVENT and TIMEX mentions given in [] from document."},
                                        {"role": "user", "content": f"This is the document: {example_doc['context']}."},
                                        {"role": "user", "content": f"List all mentions CONTAINS {example_doc['event_mentions'][0]['id_in_number']}. If there is no relations, return an empty array."},
                                        {"role": "assistant", "content": f"{example_answer}"},
                                        {"role": "user", "content": f"This is the document: {doc['context']}."},
                                        {"role": "user", "content": f"List all mentions CONTAINS {mention['id_in_number']}. If there is no relations, return an empty array."},
                                    ])

                                    result["temporal_relations"][sub_relation][mention["id_in_number"]] = completion.choices[0].message["content"]
                                    break
                
                                except openai.OpenAIError as e:
                                    randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                                    sleep_dur = delay_secs + randomness_collision_avoidance
                                    print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                                    time.sleep(sleep_dur)
                                    continue

                    if sub_relation == "OVERLAP":
                        for mention in doc["all_mentions"]:
                            for delay_secs in (2**x for x in range(0, 6)):
                                try:
                                    # Call openai request such as text completion
                                    if example_doc['event_mentions'][0]['id_in_number'] in example_doc['temporal_relations_event_order'][sub_relation]:
                                        example_answer = example_doc['temporal_relations_event_order'][sub_relation][example_doc['event_mentions'][0]['id_in_number']]
                                    else:
                                        example_answer = []
                                    completion = openai.ChatCompletion.create(
                                    model="gpt-4-1106-preview",
                                    messages=[
                                        {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given in [] after triggering words. All predictions should be an array with elements being EVENT and TIMEX mentions given in [] from document."},
                                        {"role": "user", "content": f"This is the document: {example_doc['context']}."},
                                        {"role": "user", "content": f"List all mentions OVERLAP {example_doc['event_mentions'][0]['id_in_number']}. If there is no relations, return an empty array."},
                                        {"role": "assistant", "content": f"{example_answer}"},
                                        {"role": "user", "content": f"This is the document: {doc['context']}."},
                                        {"role": "user", "content": f"List all mentions OVERLAP {mention['id_in_number']}. If there is no relations, return an empty array."},
                                    ])

                                    result["temporal_relations"][sub_relation][mention["id_in_number"]] = completion.choices[0].message["content"]
                                    break
                
                                except openai.OpenAIError as e:
                                    randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                                    sleep_dur = delay_secs + randomness_collision_avoidance
                                    print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                                    time.sleep(sleep_dur)
                                    continue

                    if sub_relation == "BEGINS-ON":
                        for mention in doc["all_mentions"]:
                            for delay_secs in (2**x for x in range(0, 6)):
                                try:
                                # Call openai request such as text completion
                                    if example_doc['event_mentions'][0]['id_in_number'] in example_doc['temporal_relations_event_order'][sub_relation]:
                                        example_answer = example_doc['temporal_relations_event_order'][sub_relation][example_doc['event_mentions'][0]['id_in_number']]
                                    else:
                                        example_answer = []
                                    completion = openai.ChatCompletion.create(
                                    model="gpt-4-1106-preview",
                                    messages=[
                                        {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given in [] after triggering words. All predictions should be an array with elements being EVENT and TIMEX mentions given in [] from document."},
                                        {"role": "user", "content": f"This is the document: {example_doc['context']}."},
                                        {"role": "user", "content": f"List all mentions BEGINS-ON {example_doc['event_mentions'][0]['id_in_number']}. If there is no relations, return an empty array."},
                                        {"role": "assistant", "content": f"{example_answer}"},
                                        {"role": "user", "content": f"This is the document: {doc['context']}."},
                                        {"role": "user", "content": f"List all mentions BEGINS-ON {mention['id_in_number']}. If there is no relations, return an empty array."},
                                    ])

                                    result["temporal_relations"][sub_relation][mention["id_in_number"]] = completion.choices[0].message["content"]
                                    break
                
                                except openai.OpenAIError as e:
                                    randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                                    sleep_dur = delay_secs + randomness_collision_avoidance
                                    print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                                    time.sleep(sleep_dur)
                                    continue

                    if sub_relation == "ENDS-ON":
                        for mention in doc["all_mentions"]:
                            for delay_secs in (2**x for x in range(0, 6)):
                                try:
                                # Call openai request such as text completion
                                    if example_doc['event_mentions'][0]['id_in_number'] in example_doc['temporal_relations_event_order'][sub_relation]:
                                        example_answer = example_doc['temporal_relations_event_order'][sub_relation][example_doc['event_mentions'][0]['id_in_number']]
                                    else:
                                        example_answer = []
                                    completion = openai.ChatCompletion.create(
                                    model="gpt-4-1106-preview",
                                    messages=[
                                        {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given in [] after triggering words. All predictions should be an array with elements being EVENT and TIMEX mentions given in [] from document."},
                                        {"role": "user", "content": f"This is the document: {example_doc['context']}."},
                                        {"role": "user", "content": f"List all mentions ENDS-ON {example_doc['event_mentions'][0]['id_in_number']}. If there is no relations, return an empty array."},
                                        {"role": "assistant", "content": f"{example_answer}"},
                                        {"role": "user", "content": f"This is the document: {doc['context']}."},
                                        {"role": "user", "content": f"List all mentions ENDS-ON {mention['id_in_number']}. If there is no relations, return an empty array."},
                                    ])

                                    result["temporal_relations"][sub_relation][mention["id_in_number"]] = completion.choices[0].message["content"]
                                    break
                
                                except openai.OpenAIError as e:
                                    randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                                    sleep_dur = delay_secs + randomness_collision_avoidance
                                    print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                                    time.sleep(sleep_dur)
                                    continue

                    if sub_relation == "SIMULTANEOUS":
                        for mention in doc["all_mentions"]:
                            for delay_secs in (2**x for x in range(0, 6)):
                                try:
                                # Call openai request such as text completion
                                    if example_doc['event_mentions'][0]['id_in_number'] in example_doc['temporal_relations_event_order'][sub_relation]:
                                        example_answer = example_doc['temporal_relations_event_order'][sub_relation][example_doc['event_mentions'][0]['id_in_number']]
                                    else:
                                        example_answer = []
                                    completion = openai.ChatCompletion.create(
                                    model="gpt-4-1106-preview",
                                    messages=[
                                        {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given in [] after triggering words. All predictions should be an array with elements being EVENT and TIMEX mentions given in [] from document."},
                                        {"role": "user", "content": f"This is the document: {example_doc['context']}."},
                                        {"role": "user", "content": f"List all mentions happened SIMULTANEOUS {example_doc['event_mentions'][0]['id_in_number']}. If there is no relations, return an empty array."},
                                        {"role": "assistant", "content": f"{example_answer}"},
                                        {"role": "user", "content": f"This is the document: {doc['context']}."},
                                        {"role": "user", "content": f"List all mentions happened SIMULTANEOUS {mention['id_in_number']}. If there is no relations, return an empty array."},
                                    ])

                                    result["temporal_relations"][sub_relation][mention["id_in_number"]] = completion.choices[0].message["content"]
                                    break
                
                                except openai.OpenAIError as e:
                                    randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                                    sleep_dur = delay_secs + randomness_collision_avoidance
                                    print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                                    time.sleep(sleep_dur)
                                    continue
                print(f"Doc #{i} temporal result: {result['temporal_relations']}")
            elif relation == "causal":
                for sub_relation in ["CAUSE", "PRECONDITION"]:
                    if sub_relation == "CAUSE":
                        for mention in doc["all_mentions"]:
                            for delay_secs in (2**x for x in range(0, 6)):
                                try:
                                # Call openai request such as text completion
                                    if example_doc['event_mentions'][0]['id_in_number'] in example_doc['causal_relations_event_order'][sub_relation]:
                                        example_answer = example_doc['causal_relations_event_order'][sub_relation][example_doc['event_mentions'][0]['id_in_number']]
                                    else:
                                        example_answer = []
                                    completion = openai.ChatCompletion.create(
                                    model="gpt-4-1106-preview",
                                    messages=[
                                        {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given in [] after triggering words. All predictions should be an array with elements being EVENT and TIMEX mentions given in [] from document."},
                                        {"role": "user", "content": f"This is the document: {example_doc['context']}."},
                                        {"role": "user", "content": f"List all mentions CAUSE {example_doc['event_mentions'][0]['id_in_number']}. If there is no relations, return an empty array."},
                                        {"role": "assistant", "content": f"{example_answer}"},
                                        {"role": "user", "content": f"This is the document: {doc['context']}."},
                                        {"role": "user", "content": f"List all mentions CAUSE {mention['id_in_number']}. If there is no relations, return an empty array."},
                                    ])

                                    result["causal_relations"][sub_relation][mention["id_in_number"]] = completion.choices[0].message["content"]
                                    break
                
                                except openai.OpenAIError as e:
                                    randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                                    sleep_dur = delay_secs + randomness_collision_avoidance
                                    print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                                    time.sleep(sleep_dur)
                                    continue
                    if sub_relation == "PRECONDITION":
                        for mention in doc["all_mentions"]:
                            for delay_secs in (2**x for x in range(0, 6)):
                                try:
                                # Call openai request such as text completion
                                    if example_doc['event_mentions'][0]['id_in_number'] in example_doc['causal_relations_event_order'][sub_relation]:
                                        example_answer = example_doc['causal_relations_event_order'][sub_relation][example_doc['event_mentions'][0]['id_in_number']]
                                    else:
                                        example_answer = []
                                    completion = openai.ChatCompletion.create(
                                    model="gpt-4-1106-preview",
                                    messages=[
                                        {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given in [] after triggering words. All predictions should be an array with elements being EVENT and TIMEX mentions given in [] from document."},
                                        {"role": "user", "content": f"This is the document: {example_doc['context']}."},
                                        {"role": "user", "content": f"List all mentions that are PRECONDITION of {example_doc['event_mentions'][0]['id_in_number']}. If there is no relations, return an empty array."},
                                        {"role": "assistant", "content": f"{example_answer}"},
                                        {"role": "user", "content": f"This is the document: {doc['context']}."},
                                        {"role": "user", "content": f"List all mentions that are PRECONDITION of {mention['id_in_number']}. If there is no relations, return an empty array."},
                                    ])

                                    result["causal_relations"][sub_relation][mention["id_in_number"]] = completion.choices[0].message["content"]
                                    break
                
                                except openai.OpenAIError as e:
                                    randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                                    sleep_dur = delay_secs + randomness_collision_avoidance
                                    print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                                    time.sleep(sleep_dur)
                                    continue
                print(f"Doc #{i} causal result: {result['causal_relations']}")
            
            elif relation == "subevent":
                for mention in doc["all_mentions"]:
                    for delay_secs in (2**x for x in range(0, 6)):
                        try:
                        # Call openai request such as text completion
                            if example_doc['event_mentions'][0]['id_in_number'] in example_doc['subevent_relations_event_order']:
                                example_answer = example_doc['subevent_relations_event_order'][example_doc['event_mentions'][0]['id_in_number']]
                            else:
                                example_answer = []
                            completion = openai.ChatCompletion.create(
                            model="gpt-4-1106-preview",
                            messages=[
                                {"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given in [] after triggering words. All predictions should be an array with elements being EVENT and TIMEX mentions given in [] from document."},
                                {"role": "user", "content": f"This is the document: {example_doc['context']}."},
                                {"role": "user", "content": f"List all mentions that are SUBEVENT of {example_doc['event_mentions'][0]['id_in_number']}. If there is no relations, return an empty array."},
                                {"role": "assistant", "content": f"{example_answer}"},
                                {"role": "user", "content": f"This is the document: {doc['context']}."},
                                {"role": "user", "content": f"List all mentions that are SUBEVENT of {mention['id_in_number']}. If there is no relations, return an empty array."},
                            ])
                                
                            result["subevent_relations"][mention["id_in_number"]] = completion.choices[0].message["content"]
                            break
        
                        except openai.OpenAIError as e:
                            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
                            sleep_dur = delay_secs + randomness_collision_avoidance
                            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
                            time.sleep(sleep_dur)
                            continue
                print(f"Doc #{i} subevent result: {result['subevent_relations']}")
        
        with open(os.path.join(output_dir, "test_prediction.jsonl"), "a")as f:
            f.write(json.dumps(result))
            f.write("\n")