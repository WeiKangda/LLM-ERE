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
        output_dir = "./output/llama2_bulk_prediction/test"
    else:
        output_dir = "./output/llama2_bulk_prediction/valid"
    data_dir = "../data/MAVEN_ERE"

    if os.path.exists(os.path.join(output_dir, "log.txt")):
        os.remove(os.path.join(output_dir, "log.txt"))
    sys.stdout = open(os.path.join(output_dir, "log.txt"), 'w')

    if os.path.exists(os.path.join(output_dir, "test_prediction.jsonl")) and not resume_eval:
        os.remove(os.path.join(output_dir, "test_prediction.jsonl"))

    resume_doc_number = 0
    if resume_eval:
        with open(os.path.join(output_dir, f"test_prediction.jsonl"), "r")as f:
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
                doc["event_mentions"] = []
                #mention_in_sent_count = defaultdict(int)
                mention_in_sent = defaultdict(list)
                eventnumberid2mentionid = {}
                doc["corefernce_relations"] = []

                for event in doc["events"]:
                    for mention in event["mention"]:
                        doc["event_mentions"].append(mention)
                doc["event_mentions"] = sorted(doc["event_mentions"], key=lambda x: (x["sent_id"], x["offset"][0]))

                id_in_event_number = 0 
                for mention in doc["event_mentions"]:
                    num = 0
                    for previous_mention in mention_in_sent[mention["sent_id"]]:
                        if previous_mention["offset"][1] < mention["offset"][1]:
                            num += 1
                    mention["offset"] = [mention["offset"][0] + num, mention["offset"][1] + num]
                    mention["id_in_event_number"] = f"Event_{id_in_event_number}"
                    mention_in_sent[mention["sent_id"]].append(mention)
                    
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
                    eventnumberid2mentionid[mention["id_in_timex_number"]] = mention["id"]
                    id_in_timex_number += 1 
                
                doc["context"] = doc["tokens"]
                for mention in doc["event_mentions"]:
                    doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id_in_event_number']}]")
                for mention in doc["TIMEX"]:
                    doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id_in_timex_number']}]")
                doc["context"] = [' '.join(join_punctuation(sentence)) for sentence in doc["context"]]
                doc["context"] = ' '.join(doc["context"])
                #print(doc["context"])
                all_eventnumberid2mentionid[doc["id"]] = eventnumberid2mentionid

                docs.append(doc)
    else:
        with open(os.path.join(data_dir, f"test.jsonl"))as f:
            lines = f.readlines()
            for i, line in enumerate(tqdm(lines, desc="Loading test data")):
                if resume_eval and i < resume_doc_number:
                    continue
                doc = json.loads(line.strip())
                #doc["event_mentions"] = []
                #mention_in_sent_count = defaultdict(int)
                mention_in_sent = defaultdict(list)
                eventnumberid2mentionid = {}
                doc["corefernce_relations"] = []

                doc["event_mentions"] = sorted(doc["event_mentions"], key=lambda x: (x["sent_id"], x["offset"][0]))

                id_in_event_number = 0 
                for mention in doc["event_mentions"]:
                    num = 0
                    for previous_mention in mention_in_sent[mention["sent_id"]]:
                        if previous_mention["offset"][1] < mention["offset"][1]:
                            num += 1
                    mention["offset"] = [mention["offset"][0] + num, mention["offset"][1] + num]
                    mention["id_in_event_number"] = f"Event_{id_in_event_number}"
                    mention_in_sent[mention["sent_id"]].append(mention)
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
                    eventnumberid2mentionid[mention["id_in_timex_number"]] = mention["id"]
                    id_in_timex_number += 1 
                
                doc["context"] = doc["tokens"]
                for mention in doc["event_mentions"]:
                    doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id_in_event_number']}]")
                for mention in doc["TIMEX"]:
                    doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id_in_timex_number']}]")
                doc["context"] = [' '.join(join_punctuation(sentence)) for sentence in doc["context"]]
                doc["context"] = ' '.join(doc["context"])
                #print(doc["context"])
                all_eventnumberid2mentionid[doc["id"]] = eventnumberid2mentionid

                docs.append(doc)

    if not resume_eval:
        with open(os.path.join(output_dir, "eventnumberid2mentionid.jsonl"), "w") as outfile:
            json.dump(all_eventnumberid2mentionid, outfile)

    example_doc = None
    with open(f"./example.jsonl")as f:
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
            mention["id_in_event_number"] = f"Event_{id_in_event_number}"
            mention_in_sent[mention["sent_id"]].append(mention)
            mentionid2eventnumberid[mention["id"]] = mention["id_in_event_number"]
            id_in_event_number += 1
        
        example_doc["TIMEX"] = sorted(example_doc["TIMEX"], key=lambda x: (x["sent_id"], x["offset"][0]))

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
            example_doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id_in_event_number']}]")
        for mention in example_doc["TIMEX"]:
            example_doc["context"][mention["sent_id"]].insert(mention["offset"][1], f"[{mention['id_in_timex_number']}]")
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
        
        subevent_temp = []
        subevent_order = []
        for pair in example_doc["subevent_relations"]:
            new_pair = []
            for event in pair:
                new_pair.append(mentionid2eventnumberid[eventid2mentionid[event][0]])
            subevent_temp.append(new_pair)
            subevent_order.append(mentions_in_order_dict[new_pair[1]])
        example_doc["subevent_relations"] = sort_list(subevent_temp, subevent_order)

        coref_temp = []
        coref_order = []
        for pair in example_doc["corefernce_relations"]:
            new_pair = []
            for event in pair:
                new_pair.append(mentionid2eventnumberid[event])
            coref_temp.append(new_pair)
            coref_order.append(mentions_in_order_dict[new_pair[1]])
        example_doc["corefernce_relations"] = sort_list(coref_temp, coref_order)
        #print(example_doc)
        #exit()

    for doc in tqdm(docs, desc="Predicting"):
        result = {"id": doc["id"], "coreference": None, "causal_relations": {}, "temporal_relations": {}, "subevent_relations": None}
        for relation in ["coreference", "temporal", "causal", "subevent"]:
            if relation == "coreference":
                # Call openai request such as text completion
                dialogs: List[Dialog] = [[{"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given. Coreference and subevent relations are binary. For temporal relations, there are 6 types: BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, and SIMULTANEOUS. For causal relations, there are 2 types: CAUSE and PRECONDITION. Note that the order of the events matter. SIMULTANEOUS and BEGINS-ON are bidirectional relations. If there is no relations, return an empty array."},
                    {"role": "user", "content": f"This is the document: {example_doc['context']}. What are the coreference relations?"},
                    {"role": "assistant", "content": f"{example_doc['corefernce_relations']}"},
                    {"role": "user", "content": f"This is the document: {doc['context']}. What are the coreference relations?"}
                ]]

                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )

                result["coreference"] = results[0]['generation']['content']
                print(f"coreference predictions: {results[0]['generation']['content']}")
            elif relation == "temporal":
                # Call openai request such as text completion
                dialogs: List[Dialog] = [[{"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given. Coreference and subevent relations are binary. For temporal relations, there are 6 types: BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, and SIMULTANEOUS. For causal relations, there are 2 types: CAUSE and PRECONDITION. Note that the order of the events matter. SIMULTANEOUS and BEGINS-ON are bidirectional relations. If there is no relations, return an empty array. You should always  finish the answer instead of using '...'."},
                    {"role": "user", "content": f"This is the document: {example_doc['context']}. What are the temporal relations? Make sure to finish your answer."},
                    {"role": "assistant", "content": f"{example_doc['temporal_relations']}"},
                    {"role": "user", "content": f"This is the document: {doc['context']}. What are the temporal relations? Make sure to finish your answer."}
                ]]

                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )

                result["temporal_relations"] = results[0]['generation']['content']
                print(f"temporal predictions: {results[0]['generation']['content']}")
            elif relation == "causal":
                # Call openai request such as text completion
                dialogs: List[Dialog] = [[{"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given. Coreference and subevent relations are binary. For temporal relations, there are 6 types: BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, and SIMULTANEOUS. For causal relations, there are 2 types: CAUSE and PRECONDITION. Note that the order of the events matter. SIMULTANEOUS and BEGINS-ON are bidirectional relations. If there is no relations, return an empty array. You should always  finish the answer instead of using '...'."},
                    {"role": "user", "content": f"This is the document: {example_doc['context']}. What are the causal relations? Make sure to finish your answer."},
                    {"role": "assistant", "content": f"{example_doc['causal_relations']}"},
                    {"role": "user", "content": f"This is the document: {doc['context']}. What are the causal relations? Make sure to finish your answer."}
                ]]

                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                    
                result["causal_relations"] = results[0]['generation']['content']
                print(f"causal predictions: {results[0]['generation']['content']}")
            elif relation == "subevent":
                # Call openai request such as text completion
                dialogs: List[Dialog] = [[{"role": "system", "content": "You are an anotator for the MAVEN-ERE dataset. Your task is to extract event coreference, temporal, causal, and subevent relations between event and TIMEX mentions from given documents, where all event and TIMEX mentions are given. Coreference and subevent relations are binary. For temporal relations, there are 6 types: BEFORE, CONTAINS, OVERLAP, BEGINS-ON, ENDS-ON, and SIMULTANEOUS. For causal relations, there are 2 types: CAUSE and PRECONDITION. Note that the order of the events matter. SIMULTANEOUS and BEGINS-ON are bidirectional relations. If there is no relations, return an empty array. You should always  finish the answer instead of using '...'."},
                    {"role": "user", "content": f"This is the document: {example_doc['context']}. What are the subevent relations? Make sure to finish your answer."},
                    {"role": "assistant", "content": f"{example_doc['subevent_relations']}"},
                    {"role": "user", "content": f"This is the document: {doc['context']}. What are the subevent relations? Make sure to finish your answer."}
                ]]

                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                    
                result["subevent_relations"] = results[0]['generation']['content']
                print(f"subevent predictions: {results[0]['generation']['content']}")
            
        with open(os.path.join(output_dir, "test_prediction.jsonl"), "a")as f:
            f.write(json.dumps(result))
            f.write("\n")

if __name__ == "__main__":
    fire.Fire(main)