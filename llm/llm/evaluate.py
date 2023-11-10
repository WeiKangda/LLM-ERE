import json
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import re

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
    output_dir = "./llm/output/valid"
    input_dir = "./data/MAVEN_ERE"

    answer_docs = {}
    with open(os.path.join(input_dir, f"valid_first10.jsonl"))as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines, desc="Loading answers")):
            example_doc = json.loads(line.strip())
            example_doc["event_mentions"] = []
            mention_in_sent = defaultdict(list)
            example_doc["corefernce_relations"] = []
            eventid2mentionid = defaultdict(list)
            mentionid2eventnumberid = {}
            
            id_in_event_number = 0
            for event in example_doc["events"]:
                if len(event["mention"]) != 1:
                        for i in range(len(event["mention"]) - 1):
                            example_doc["corefernce_relations"].append([event["mention"][i]["id"], event["mention"][i+1]["id"]])

                for mention in event["mention"]:
                    num = 0
                    for previous_mention in mention_in_sent[mention["sent_id"]]:
                        if previous_mention["offset"][1] < mention["offset"][1]:
                            num += 1
                    mention["offset"] = [mention["offset"][0] + num, mention["offset"][1] + num]
                    mention["id_in_event_number"] = f"Event_{id_in_event_number}"
                    mention_in_sent[mention["sent_id"]].append(mention)
                    example_doc["event_mentions"].append(mention)
                    eventid2mentionid[event["id"]].append(mention["id"])
                    mentionid2eventnumberid[mention["id"]] = mention["id_in_event_number"]
                    id_in_event_number += 1
            
            id_in_timex_number = 0 
            for mention in example_doc["TIMEX"]:
                num = 0
                for previous_mention in mention_in_sent[mention["sent_id"]]:
                    if previous_mention["offset"][1] < mention["offset"][1]:
                        num += 1
                mention["offset"] = [mention["offset"][0] + num, mention["offset"][1] + num]
                mention["id_in_timex_number"] = f"TIMEX_{id_in_timex_number}"
                mention_in_sent[mention["sent_id"]].append(mention)
                eventid2mentionid[mention["id"]].append(mention["id"])
                mentionid2eventnumberid[mention["id"]] = mention["id_in_timex_number"]
                id_in_timex_number += 1 
                
            example_doc["context"] = example_doc["tokens"]
            for mention in example_doc["event_mentions"]:
                example_doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id']}]")
            for mention in example_doc["TIMEX"]:
                example_doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id']}]")
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
                        if len(eventid2mentionid[event]) == 0:
                            continue
                        new_pair.append(eventid2mentionid[event][0])
                    if len(new_pair) == 0:
                        continue
                    temporal_temp[temporal_type].append(new_pair)
                    temporal_order[temporal_type].append(mentions_in_order_dict[new_pair[1]])
            temporal_relations = {"BEFORE": sort_list(temporal_temp["BEFORE"], temporal_order["BEFORE"]),\
                                "CONTAINS": sort_list(temporal_temp["CONTAINS"], temporal_order["CONTAINS"]),\
                                "OVERLAP": sort_list(temporal_temp["OVERLAP"], temporal_order["OVERLAP"]),\
                                "BEGINS-ON": sort_list(temporal_temp["BEGINS-ON"], temporal_order["BEGINS-ON"]),\
                                "ENDS-ON": sort_list(temporal_temp["ENDS-ON"], temporal_order["ENDS-ON"]),\
                                "SIMULTANEOUS": sort_list(temporal_temp["SIMULTANEOUS"], temporal_order["SIMULTANEOUS"])}
            example_doc["temporal_relations"] = temporal_relations

            causal_order = {"CAUSE": [], "PRECONDITION": []}
            causal_temp = {"CAUSE": [], "PRECONDITION": []}
            for causal_type in example_doc["causal_relations"]:
                for pair in example_doc["causal_relations"][causal_type]:
                    new_pair = []
                    for event in pair:
                        if len(eventid2mentionid[event]) == 0:
                            continue
                        new_pair.append(eventid2mentionid[event][0])
                    if len(new_pair) == 0:
                        continue
                    causal_temp[causal_type].append(new_pair)
                    causal_order[causal_type].append(mentions_in_order_dict[new_pair[1]])
            causal_relations = {"CAUSE": sort_list(causal_temp["CAUSE"], causal_order["CAUSE"]), \
                                "PRECONDITION": sort_list(causal_temp["PRECONDITION"], causal_order["PRECONDITION"])}
            example_doc["causal_relations"] = causal_relations
            
            subevent_temp = []
            subevent_order = []
            for pair in example_doc["subevent_relations"]:
                new_pair = []
                for event in pair:
                    if len(eventid2mentionid[event]) == 0:
                        continue
                    new_pair.append(eventid2mentionid[event][0])
                if len(new_pair) == 0:
                        continue
                subevent_temp.append(new_pair)
                subevent_order.append(mentions_in_order_dict[new_pair[1]])
            example_doc["subevent_relations"] = sort_list(subevent_temp, subevent_order)

            coref_temp = []
            coref_order = []
            for pair in example_doc["corefernce_relations"]:
                new_pair = []
                for event in pair:
                    if len(eventid2mentionid[event]) == 0:
                        continue
                    new_pair.append(eventid2mentionid[event][0])
                if len(new_pair) == 0:
                        continue
                coref_temp.append(new_pair)
                coref_order.append(mentions_in_order_dict[new_pair[1]])
            example_doc["corefernce_relations"] = sort_list(coref_temp, coref_order)
            answer_docs[example_doc["id"]] = example_doc

            #print(len(example_doc["event_mentions"]))
            #print(len(example_doc["TIMEX"]))
            all_mentions = example_doc["event_mentions"]
            all_mentions.extend(example_doc["TIMEX"])

    
    pred_docs = []
    with open(os.path.join(output_dir, f"processed_test_prediction.jsonl"))as f:
        lines = f.readlines()
        for i, line in enumerate(tqdm(lines, desc="Loading answers")):
            doc = json.loads(line.strip())
            pred_docs.append(doc)

    
    BEFORE_TP = 0
    BEFORE_FN = 0
    BEFORE_FP = 0
    CONTAINS_TP = 0
    CONTAINS_FN = 0
    CONTAINS_FP = 0
    OVERLAP_TP = 0
    OVERLAP_FN = 0
    OVERLAP_FP = 0
    BEGINS_ON_TP = 0
    BEGINS_ON_FN = 0
    BEGINS_ON_FP = 0
    ENDS_ON_TP = 0
    ENDS_ON_FN = 0
    ENDS_ON_FP = 0
    SIMULTANEOUS_TP = 0
    SIMULTANEOUS_FN = 0
    SIMULTANEOUS_FP = 0
    CAUSE_TP = 0
    CAUSE_FN = 0
    CAUSE_FP = 0
    PRECONDITION_TP = 0
    PRECONDITION_FN = 0
    PRECONDITION_FP = 0
    SUBEVENT_TP = 0
    SUBEVENT_FN = 0
    SUBEVENT_FP = 0

    for pred_doc in pred_docs:
        answer_doc = answer_docs[pred_doc["id"]]
        #print(answer_doc["temporal_relations"]["CONTAINS"])
        for relation_type in ["BEFORE", "CONTAINS", "OVERLAP", "BEGINS-ON", "ENDS-ON", "SIMULTANEOUS"]:
            gold = answer_doc["temporal_relations"][relation_type]
            pred = pred_doc["temporal_relations"][relation_type]

            for pred_pair in pred:
                FP = True
                for gold_pair in gold:
                    if pred_pair[0] == gold_pair[0] and pred_pair[1] == gold_pair[1]:
                        FP = False
                        if relation_type == "BEFORE":
                            BEFORE_TP += 1
                        elif relation_type == "CONTAINS":
                            CONTAINS_TP += 1
                        elif relation_type == "OVERLAP":
                            OVERLAP_TP += 1
                        elif relation_type == "BEGINS-ON":
                            BEGINS_ON_TP += 1
                        elif relation_type == "ENDS-ON":
                            ENDS_ON_TP += 1
                        elif relation_type == "SIMULTANEOUS":
                            SIMULTANEOUS_TP += 1
                if FP:
                    if relation_type == "BEFORE":
                        BEFORE_FP += 1
                    elif relation_type == "CONTAINS":
                        CONTAINS_FP += 1
                    elif relation_type == "OVERLAP":
                        OVERLAP_FP += 1
                    elif relation_type == "BEGINS-ON":
                        BEGINS_ON_FP += 1
                    elif relation_type == "ENDS-ON":
                        ENDS_ON_FP += 1
                    elif relation_type == "SIMULTANEOUS":
                        SIMULTANEOUS_FP += 1

            for gold_pair in gold:
                FN = True
                for pred_pair in pred:
                    if pred_pair[0] == gold_pair[0] and pred_pair[1] == gold_pair[1]:
                        FN = False
                if FN:
                    if relation_type == "BEFORE":
                        BEFORE_FN += 1
                    elif relation_type == "CONTAINS":
                        CONTAINS_FN += 1
                    elif relation_type == "OVERLAP":
                        OVERLAP_FN += 1
                    elif relation_type == "BEGINS-ON":
                        BEGINS_ON_FN += 1
                    elif relation_type == "ENDS-ON":
                        ENDS_ON_FN += 1
                    elif relation_type == "SIMULTANEOUS":
                        SIMULTANEOUS_FN += 1
        
        for relation_type in ["CAUSE", "PRECONDITION"]:
            gold = answer_doc["causal_relations"][relation_type]
            pred = pred_doc["causal_relations"][relation_type]

            for pred_pair in pred:
                FP = True
                for gold_pair in gold:
                    if pred_pair[0] == gold_pair[0] and pred_pair[1] == gold_pair[1]:
                        FP = False
                        if relation_type == "CAUSE":
                            CAUSE_TP += 1
                        elif relation_type == "PRECONDITION":
                            PRECONDITION_TP += 1
                if FP:
                    if relation_type == "CAUSE":
                        CAUSE_FP += 1
                    elif relation_type == "PRECONDITION":
                        PRECONDITION_FP += 1

            for gold_pair in gold:
                FN = True
                for pred_pair in pred:
                    if pred_pair[0] == gold_pair[0] and pred_pair[1] == gold_pair[1]:
                        FN = False
                if FN:
                    if relation_type == "CAUSE":
                        CAUSE_FN += 1
                    elif relation_type == "PRECONDITION":
                        PRECONDITION_FN += 1
        
        gold = answer_doc["subevent_relations"]
        pred = pred_doc["subevent_relations"]

        for pred_pair in pred:
            FP = True
            for gold_pair in gold:
                if pred_pair[0] == gold_pair[0] and pred_pair[1] == gold_pair[1]:
                    FP = False
                    SUBEVENT_TP += 1
            if FP:
                SUBEVENT_FP += 1

        for gold_pair in gold:
            FN = True
            for pred_pair in pred:
                if pred_pair[0] == gold_pair[0] and pred_pair[1] == gold_pair[1]:
                    FN = False
            if FN:
                SUBEVENT_FN += 1
    
    temporal_result = {"BEFORE": {}, "CONTAINS": {}, "OVERLAP": {}, "BEGINS-ON": {}, "ENDS-ON": {}, "SIMULTANEOUS": {}, "micro": {}}
    for relation_type in temporal_result:
        if relation_type == "BEFORE":
            precision = BEFORE_TP / (BEFORE_TP + BEFORE_FP) if BEFORE_TP + BEFORE_FP != 0 else 0
            recall = BEFORE_TP / (BEFORE_TP + BEFORE_FN) if BEFORE_TP + BEFORE_FN != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            temporal_result[relation_type]["precision"] = "{:.5f}".format(precision)
            temporal_result[relation_type]["recall"] = "{:.5f}".format(recall)
            temporal_result[relation_type]["f1"] = "{:.5f}".format(f1)
        elif relation_type == "CONTAINS":
            precision = CONTAINS_TP / (CONTAINS_TP + CONTAINS_FP) if CONTAINS_TP + CONTAINS_FP != 0 else 0
            recall = CONTAINS_TP / (CONTAINS_TP + CONTAINS_FN) if CONTAINS_TP + CONTAINS_FN != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            temporal_result[relation_type]["precision"] = "{:.5f}".format(precision)
            temporal_result[relation_type]["recall"] = "{:.5f}".format(recall)
            temporal_result[relation_type]["f1"] = "{:.5f}".format(f1)
        elif relation_type == "OVERLAP":
            precision = OVERLAP_TP / (OVERLAP_TP + OVERLAP_FP) if OVERLAP_TP + OVERLAP_FP != 0 else 0
            recall = OVERLAP_TP / (OVERLAP_TP + OVERLAP_FN) if OVERLAP_TP + OVERLAP_FN != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            temporal_result[relation_type]["precision"] = "{:.5f}".format(precision)
            temporal_result[relation_type]["recall"] = "{:.5f}".format(recall)
            temporal_result[relation_type]["f1"] = "{:.5f}".format(f1)
        elif relation_type == "BEGINS-ON":
            precision = BEGINS_ON_TP / (BEGINS_ON_TP + BEGINS_ON_FP) if BEGINS_ON_TP + BEGINS_ON_FP != 0 else 0
            recall = BEGINS_ON_TP / (BEGINS_ON_TP + BEGINS_ON_FN) if BEGINS_ON_TP + BEGINS_ON_FN != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            temporal_result[relation_type]["precision"] = "{:.5f}".format(precision)
            temporal_result[relation_type]["recall"] = "{:.5f}".format(recall)
            temporal_result[relation_type]["f1"] = "{:.5f}".format(f1)
        elif relation_type == "ENDS-ON":
            precision = ENDS_ON_TP / (ENDS_ON_TP + ENDS_ON_FP) if ENDS_ON_TP + ENDS_ON_FP != 0 else 0
            recall = ENDS_ON_TP / (ENDS_ON_TP + ENDS_ON_FN) if ENDS_ON_TP + ENDS_ON_FN != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            temporal_result[relation_type]["precision"] = "{:.5f}".format(precision)
            temporal_result[relation_type]["recall"] = "{:.5f}".format(recall)
            temporal_result[relation_type]["f1"] = "{:.5f}".format(f1)
        elif relation_type == "SIMULTANEOUS":
            precision = SIMULTANEOUS_TP / (SIMULTANEOUS_TP + SIMULTANEOUS_FP) if SIMULTANEOUS_TP + SIMULTANEOUS_FP != 0 else 0
            recall = SIMULTANEOUS_TP / (SIMULTANEOUS_TP + SIMULTANEOUS_FN) if SIMULTANEOUS_TP + SIMULTANEOUS_FN != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            temporal_result[relation_type]["precision"] = "{:.5f}".format(precision)
            temporal_result[relation_type]["recall"] = "{:.5f}".format(recall)
            temporal_result[relation_type]["f1"] = "{:.5f}".format(f1)
        elif relation_type == "micro":
            precision = (BEFORE_TP + CONTAINS_TP + OVERLAP_TP + BEGINS_ON_TP + ENDS_ON_TP + SIMULTANEOUS_TP) / \
                        (BEFORE_TP + CONTAINS_TP + OVERLAP_TP + BEGINS_ON_TP + ENDS_ON_TP + SIMULTANEOUS_TP + \
                        BEFORE_FP + CONTAINS_FP + OVERLAP_FP + BEGINS_ON_FP + ENDS_ON_FP + SIMULTANEOUS_FP) \
                        if (BEFORE_TP + CONTAINS_TP + OVERLAP_TP + BEGINS_ON_TP + ENDS_ON_TP + SIMULTANEOUS_TP + \
                        BEFORE_FP + CONTAINS_FP + OVERLAP_FP + BEGINS_ON_FP + ENDS_ON_FP + SIMULTANEOUS_FP) != 0 else 0
            recall = (BEFORE_TP + CONTAINS_TP + OVERLAP_TP + BEGINS_ON_TP + ENDS_ON_TP + SIMULTANEOUS_TP) / \
                    (BEFORE_TP + CONTAINS_TP + OVERLAP_TP + BEGINS_ON_TP + ENDS_ON_TP + SIMULTANEOUS_TP + \
                    BEFORE_FN + CONTAINS_FN + OVERLAP_FN + BEGINS_ON_FN + ENDS_ON_FN + SIMULTANEOUS_FN) \
                    if (BEFORE_TP + CONTAINS_TP + OVERLAP_TP + BEGINS_ON_TP + ENDS_ON_TP + SIMULTANEOUS_TP + \
                    BEFORE_FN + CONTAINS_FN + OVERLAP_FN + BEGINS_ON_FN + ENDS_ON_FN + SIMULTANEOUS_FN) != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            temporal_result[relation_type]["precision"] = "{:.5f}".format(precision)
            temporal_result[relation_type]["recall"] = "{:.5f}".format(recall)
            temporal_result[relation_type]["f1"] = "{:.5f}".format(f1)
        

    causal_result = {"CAUSE": {}, "PRECONDITION": {}, "micro": {}}
    for relation_type in causal_result:
        if relation_type == "CAUSE":
            precision = CAUSE_TP / (CAUSE_TP + CAUSE_FP) if CAUSE_TP + CAUSE_FP != 0 else 0
            recall = CAUSE_TP / (CAUSE_TP + CAUSE_FN) if CAUSE_TP + CAUSE_FN != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            causal_result[relation_type]["precision"] = "{:.5f}".format(precision)
            causal_result[relation_type]["recall"] = "{:.5f}".format(recall)
            causal_result[relation_type]["f1"] = "{:.5f}".format(f1)
        elif relation_type == "PRECONDITION":
            precision = PRECONDITION_TP / (PRECONDITION_TP + PRECONDITION_FP) if PRECONDITION_TP + PRECONDITION_FP != 0 else 0
            recall = PRECONDITION_TP / (PRECONDITION_TP + PRECONDITION_FN) if PRECONDITION_TP + PRECONDITION_FN != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            causal_result[relation_type]["precision"] = "{:.5f}".format(precision)
            causal_result[relation_type]["recall"] = "{:.5f}".format(recall)
            causal_result[relation_type]["f1"] = "{:.5f}".format(f1)
        elif relation_type == "micro":
            precision = (CAUSE_TP + PRECONDITION_TP) / (CAUSE_TP + PRECONDITION_TP + CAUSE_FP + PRECONDITION_FP) \
                        if (CAUSE_TP + PRECONDITION_TP + CAUSE_FP + PRECONDITION_FP) != 0 else 0
            recall = (CAUSE_TP + PRECONDITION_TP) / (CAUSE_TP + PRECONDITION_TP + CAUSE_FN + PRECONDITION_FN) \
                    if (CAUSE_TP + PRECONDITION_TP + CAUSE_FN + PRECONDITION_FN) != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            causal_result[relation_type]["precision"] = "{:.5f}".format(precision)
            causal_result[relation_type]["recall"] = "{:.5f}".format(recall)
            causal_result[relation_type]["f1"] = "{:.5f}".format(f1)
    
    subevent_result = {"SUBEVENT": {}, "micro": {}}
    for relation_type in subevent_result:
        if relation_type == "SUBEVENT":
            precision = SUBEVENT_TP / (SUBEVENT_TP + SUBEVENT_FP) if SUBEVENT_TP + SUBEVENT_FP != 0 else 0
            recall = SUBEVENT_TP / (SUBEVENT_TP + SUBEVENT_FN) if SUBEVENT_TP + SUBEVENT_FN != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            subevent_result[relation_type]["precision"] = "{:.5f}".format(precision)
            subevent_result[relation_type]["recall"] = "{:.5f}".format(recall)
            subevent_result[relation_type]["f1"] = "{:.5f}".format(f1)
        if relation_type == "micro":
            precision = SUBEVENT_TP / (SUBEVENT_TP + SUBEVENT_FP) if SUBEVENT_TP + SUBEVENT_FP != 0 else 0
            recall = SUBEVENT_TP / (SUBEVENT_TP + SUBEVENT_FN) if SUBEVENT_TP + SUBEVENT_FN != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
            subevent_result[relation_type]["precision"] = "{:.5f}".format(precision)
            subevent_result[relation_type]["recall"] = "{:.5f}".format(recall)
            subevent_result[relation_type]["f1"] = "{:.5f}".format(f1)

    print(f"Temporal Result: {temporal_result}")
    print(f"Causal Result: {causal_result}")
    print(f"Subevent Result: {subevent_result}")
    
    if os.path.exists(os.path.join(output_dir, "result.jsonl")):
        os.remove(os.path.join(output_dir, "result.jsonl"))
    with open(os.path.join(output_dir, "result.jsonl"), "a")as f:
        f.write(json.dumps(temporal_result))
        f.write("\n")
        f.write(json.dumps(causal_result))
        f.write("\n")
        f.write(json.dumps(subevent_result))
        f.write("\n")