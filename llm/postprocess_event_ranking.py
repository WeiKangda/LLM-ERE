import json
import os

def postprocess(raw_prediction):
    with open(os.path.join(output_dir, "eventnumberid2mentionid.jsonl"), "r")as f:
        lines = f.readlines()
        eventnumberid2mentionid = json.loads(lines[0].strip())
    processed_prediction = {"id": None, "coreference": None, "causal_relations": {"CAUSE": None, "PRECONDITION": None}, "temporal_relations": {"BEFORE": None, "CONTAINS": None, "OVERLAP": None, "BEGINS-ON": None, "ENDS-ON": None, "SIMULTANEOUS": None}, "subevent_relations": None}
    processed_prediction["id"] = raw_prediction["id"]

    eventnumberid2mentionid = eventnumberid2mentionid[processed_prediction["id"]]

    temporal_temp = {"BEFORE": [], "CONTAINS": [], "OVERLAP": [], "BEGINS-ON": [], "ENDS-ON": [], "SIMULTANEOUS": []}
    for temporal_type in raw_prediction["temporal_relations"]:
        for mention_j in raw_prediction["temporal_relations"][temporal_type]:
            pred_string = raw_prediction["temporal_relations"][temporal_type][mention_j]
            pred_string = pred_string.replace("[", "")
            pred_string = pred_string.replace("]", "")
            pred_string = pred_string.replace("'", "")
            pred_array = pred_string.split(", ")
            #print(pred_array)

            for mention_i in pred_array:
                new_pair = []
                if mention_i not in eventnumberid2mentionid or mention_j not in eventnumberid2mentionid:
                    continue
                if eventnumberid2mentionid[mention_i] == eventnumberid2mentionid[mention_j]:
                    continue
                new_pair.extend([eventnumberid2mentionid[mention_i], eventnumberid2mentionid[mention_j]])
                if len(new_pair) != 2:
                    continue
                temporal_temp[temporal_type].append(new_pair)
    processed_prediction["temporal_relations"] = temporal_temp
    #print(temporal_temp)
    
    causal_temp = {"CAUSE": [], "PRECONDITION": []}
    for causal_type in raw_prediction["causal_relations"]:
        for mention_j in raw_prediction["causal_relations"][causal_type]:
            pred_string = raw_prediction["causal_relations"][causal_type][mention_j]
            pred_string = pred_string.replace("[", "")
            pred_string = pred_string.replace("]", "")
            pred_string = pred_string.replace("'", "")
            pred_array = pred_string.split(", ")
            #print(pred_array)
            if mention_j.startswith("TIME"):
                continue
            new_pair = []
            for mention_i in pred_array:
                if mention_i not in eventnumberid2mentionid or mention_j not in eventnumberid2mentionid:
                    continue
                if eventnumberid2mentionid[mention_i] == eventnumberid2mentionid[mention_j]:
                    continue
                new_pair.extend([eventnumberid2mentionid[mention_i], eventnumberid2mentionid[mention_j]])
                if len(new_pair) != 2:
                    continue
                causal_temp[causal_type].append(new_pair)
    processed_prediction["causal_relations"] = causal_temp
    #print(causal_temp)
    
    subevent_temp = []
    for mention_j in raw_prediction["subevent_relations"]:
        pred_string = raw_prediction["subevent_relations"][mention_j]
        pred_string = pred_string.replace("[", "")
        pred_string = pred_string.replace("]", "")
        pred_string = pred_string.replace("'", "")
        pred_array = pred_string.split(", ")
        if mention_j.startswith("TIME"):
                continue
        new_pair = []
        for mention_i in pred_array:
            if mention_i not in eventnumberid2mentionid or mention_j not in eventnumberid2mentionid:
                continue
            if eventnumberid2mentionid[mention_i] == eventnumberid2mentionid[mention_j]:
                continue
            new_pair.extend([eventnumberid2mentionid[mention_i], eventnumberid2mentionid[mention_j]])
            if len(new_pair) != 2:
                continue
            subevent_temp.append(new_pair)
    processed_prediction["subevent_relations"] = subevent_temp

    coref = []
    for event in raw_prediction["coreference"]:
        pred_string = raw_prediction["coreference"][event]
        pred_string = pred_string.replace("[", "")
        pred_string = pred_string.replace("]", "")
        pred_string = pred_string.replace("'", "")
        pred_array = pred_string.split(", ")
        for e in pred_array:
            if e == "":
                continue
            coref.append([e, event])
    print(coref)
    coref_temp = []
    appeared = {}
    for pair in coref:
        #cluster = list(set(cluster))
        new_cluster = []
        for i, event in enumerate(pair):
            if event not in eventnumberid2mentionid:
                continue
            if event in appeared:
                for j, cluster in enumerate(coref_temp):
                    if event in cluster:
                        coref_temp[j].append(eventnumberid2mentionid(pair[0]))
                        coref_temp[j].append(eventnumberid2mentionid(pair[1]))
                        appeared[pair[0]] = eventnumberid2mentionid[pair[0]]
                        appeared[pair[1]] = eventnumberid2mentionid[pair[1]]
                continue
            if event.startswith("TIME"):
                continue
            new_cluster.append(eventnumberid2mentionid[event])
            appeared[event] = eventnumberid2mentionid[event]
        coref_temp.append(new_cluster)

    for mention in eventnumberid2mentionid:
        #if mention in appeared:
        #        continue
        already_in_cluster = False
        for cluster in raw_prediction["coreference"]:
            if mention in cluster or mention.startswith("TIME"):
                already_in_cluster = True
        if not already_in_cluster:
            coref_temp.append([eventnumberid2mentionid[mention]])
            appeared[mention] = eventnumberid2mentionid[mention]
    coreference_temp =[]
    for i, cluster in enumerate(coref_temp):
        if len(cluster) == 0:
            continue
        cluster = list(set(cluster))
        coreference_temp.append(cluster)
    processed_prediction["coreference"] = coreference_temp
    print(coreference_temp)
    return processed_prediction

if __name__ == "__main__":
    output_dir = "./llm/output/gpt2/valid"

    processed_predictions = []
    with open(os.path.join(output_dir, "test_prediction.jsonl"), "r")as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            print(f"Doc #{i}")
            
            raw_prediction = json.loads(line.strip())
            processed_predictions.append(postprocess(raw_prediction))
    
    if os.path.exists(os.path.join(output_dir, "processed_test_prediction.jsonl")):
        os.remove(os.path.join(output_dir, "processed_test_prediction.jsonl"))

    with open(os.path.join(output_dir, "processed_test_prediction.jsonl"), "w")as f:
        f.writelines("\n".join([json.dumps(processed_prediction) for processed_prediction in processed_predictions]))