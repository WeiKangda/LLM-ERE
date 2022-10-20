# %%
import json
from torch.utils.data import DataLoader, Dataset
import torch
import os
from tqdm import tqdm
# from utils import flatten
import nltk

def valid_split(point, spans):
    # retain context of at least 3 tokens
    for sp in spans:
        if point > sp[0] - 3 and point <= sp[1] + 3:
            return False
    return True

def split_spans(point, spans):
    part1 = []
    part2 = []
    i = 0
    for sp in spans:
        if sp[1] < point:
            part1.append(sp)
            i += 1
        else:
            break
    part2 = spans[i:]
    return part1, part2

def tokenize_sent(text):
    return nltk.sent_tokenize(text)

# def type_tokens(type_str):
#     return [f"<{type_str}>", f"<{type_str}/>"]

class Document:
    def __init__(self, data):
        self.id = data["id"]
        self.text = tokenize_sent(data["sentence"])
        # self.merged_events = data["merged_events"]
        self.events = data["events"]
        
        # self.flatten()
        self.populate_event_spans()
    
    def event_id(self, event):
        return event["trigger_word"] + "_" + str(event["position"][0])
    
    def add_sent_id(self, events):
        new_events = []
        e_id = 0
        cur_len = 0
        for i, sent in enumerate(self.text):
            old_len = cur_len
            cur_len += len(sent) + 1
            while e_id < len(events) and events[e_id]["position"][0] < cur_len:
                e = events[e_id]
                e["sent_id"] = i
                e["position"] = [e["position"][0]-old_len, e["position"][1]-old_len] # adjust position in sentence
                assert sent[e["position"][0]:e["position"][1]] == e["trigger_word"]
                new_events.append(e)
                e_id += 1
        return new_events
    
    def populate_event_spans(self):
        # lengths = [len(sent) for sent in self.words]
        event_mentions = [] # {"id": int, "trigger_word": str, "position":[],}
        merged_events = []
        for event in self.events:
            for item in event["triggers"]:
                item["event_type"] = event["type"]
            merged_events.append(event["triggers"])
            event_mentions.extend(event["triggers"])
        events = sorted(event_mentions, key=lambda x: x["position"][0])
        event2id = {self.event_id(e):idx for idx, e in enumerate(events)} # sorted events to index

        self.label_groups = [[event2id[self.event_id(e)] for e in events] for events in merged_events] # List[List[int]] each sublist is a group of event index that co-references each other

        events = self.add_sent_id(events)
        self.sorted_event_spans = [(event["sent_id"], event["position"], event["event_type"]) for event in events]


class myDataset(Dataset):
    def __init__(self, tokenizer, data_dir, split, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.load_examples(data_dir, split)
        self.examples = self.examples
        self.tokenize()
        self.to_tensor()
    
    def load_examples(self, data_dir, split):
        self.examples = []
        with open(os.path.join(data_dir, f"{split}.jsonl"))as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line.strip())
            doc = Document(data)
            if doc.sorted_event_spans:
                self.examples.append(doc)
    
    def tokenize(self):
        # {input_ids, event_spans, event_group}
        # TODO: split articless into part of max_length
        self.tokenized_samples = []
        for example in tqdm(self.examples, desc="tokenizing"):
            event_spans = [] # [[(start, end)], [],...]
            input_ids = [] # [[], [], ...]

            label_groups = example.label_groups
            spans = example.sorted_event_spans
            text = example.text
            event_id = 0
            sub_input_ids = [self.tokenizer.cls_token_id]
            sub_event_spans = []
            for sent_id, word in enumerate(text):
                i = 0
                tmp_event_spans = []
                tmp_input_ids = []
                # add special tokens for event
                while event_id < len(spans) and spans[event_id][0] == sent_id:
                    sp = spans[event_id]
                    if i < sp[1][0]:
                        context_ids = self.tokenizer(word[i:sp[1][0]], add_special_tokens=False)["input_ids"]
                        tmp_input_ids += context_ids
                    event_ids = self.tokenizer(word[sp[1][0]:sp[1][1]], add_special_tokens=False)["input_ids"]
                    start = len(tmp_input_ids)
                    end = len(tmp_input_ids) + len(event_ids)
                    tmp_event_spans.append((start, end))
                    # special_ids = self.tokenizer(type_tokens(sp[2]), is_split_into_words=True, add_special_tokens=False)["input_ids"]
                    # assert len(special_ids) == 2, print(f"special tokens <{sp[2]}> and <{sp[2]}/> may not be added to tokenizer.")
                    tmp_input_ids += event_ids

                    i = sp[1][1]
                    event_id += 1
                if word[i:]:
                    tmp_input_ids += self.tokenizer(word[i:],add_special_tokens=False)["input_ids"]

                # TODO add sep token
                tmp_input_ids += [self.tokenizer.sep_token_id]
                
                if len(sub_input_ids) + len(tmp_input_ids) <= self.max_length:
                    # print(len(sub_input_ids) + len(tmp_input_ids))
                    sub_event_spans += [(sp[0]+len(sub_input_ids), sp[1]+len(sub_input_ids)) for sp in tmp_event_spans]
                    sub_input_ids += tmp_input_ids
                else:
                    # print("exceed max length! truncate")
                    assert len(sub_input_ids) <= self.max_length
                    input_ids.append(sub_input_ids)
                    event_spans.append(sub_event_spans)

                    # assert len(tmp_input_ids) < self.max_length, print("A sentence too long!\n %s" % " ".join(words[sent_id])) # 3580:
                    while len(tmp_input_ids) >= self.max_length:
                        split_point = self.max_length - 1
                        while not valid_split(split_point, tmp_event_spans):
                            split_point -= 1
                        tmp_event_spans_part1, tmp_event_spans = split_spans(split_point, tmp_event_spans)
                        tmp_input_ids_part1, tmp_input_ids = tmp_input_ids[:split_point], tmp_input_ids[split_point:]

                        input_ids.append([self.tokenizer.cls_token_id] + tmp_input_ids_part1)
                        event_spans.append([(sp[0]+1, sp[1]+1) for sp in tmp_event_spans_part1])

                        tmp_event_spans = [(sp[0]-len(tmp_input_ids_part1), sp[1]-len(tmp_input_ids_part1)) for sp in tmp_event_spans]
                        # sub_input_ids = [self.tokenizer.cls_token_id] + tmp_input_ids_part2

                    sub_event_spans = [(sp[0]+1, sp[1]+1) for sp in tmp_event_spans]
                    sub_input_ids = [self.tokenizer.cls_token_id] + tmp_input_ids
            if sub_input_ids:
                input_ids.append(sub_input_ids)
                event_spans.append(sub_event_spans)
            
            assert event_id == len(spans)
                
            tokenized = {"input_ids": input_ids, "attention_mask": None, "event_spans": event_spans, "label_groups": label_groups}
            self.tokenized_samples.append(tokenized)
    
    def to_tensor(self):
        for item in self.tokenized_samples:
            # print(item)
            attention_mask = []
            for ids in item["input_ids"]:
                mask = [1] * len(ids)
                while len(ids) < self.max_length:
                    ids.append(self.tokenizer.pad_token_id)
                    mask.append(0)
                attention_mask.append(mask)
            item["input_ids"] = torch.LongTensor(item["input_ids"])
            item["attention_mask"] = torch.LongTensor(attention_mask)
            # retain_event_index = [i for i in range(len(item["event_spans"])) if item["event_spans"][i][1] < self.max_length]
            # item["event_spans"] = [item["event_spans"][i] for i in retain_event_index]
            # item["label_groups"] = [[i for i in gr if i in retain_event_index] for gr in item["label_groups"]]
    
    def __getitem__(self, index):
        return self.tokenized_samples[index]

    def __len__(self):
        return len(self.tokenized_samples)


def collator(data):
    collate_data = {"input_ids": [], "attention_mask": [], "event_spans": [], "label_groups": [], "splits": [0]}
    for d in data:
        for k in d:
            collate_data[k].append(d[k])
    lengths = [ids.size(0) for ids in collate_data["input_ids"]]
    for l in lengths:
        collate_data["splits"].append(collate_data["splits"][-1]+l)


    collate_data["input_ids"] = torch.cat(collate_data["input_ids"])
    collate_data["attention_mask"] = torch.cat(collate_data["attention_mask"])
    return collate_data

def get_dataloader(tokenizer, split, data_dir="../data/processed/ACE", max_length=128, batch_size=8, shuffle=True, sample_rate=None):
    dataset = myDataset(tokenizer, data_dir, split, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)

if __name__ == "__main__":
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # tokens = get_special_tokens()
    # print(tokens)
    # tokenizer.add_special_tokens(tokens)
    # n = tokenizer.add_tokens(tokens)
    # print(n)
    # dataset = myDataset(tokenizer, "../data/", "test")
    # print(dataset[0])
    # print(dataset[1])
    # print(dataset[2])
    dataloader = get_dataloader(tokenizer, "test", data_dir="../../data/processed/ACE", shuffle=False, max_length=512, batch_size=1, sample_rate=None)
    for data in dataloader:
        print(data["input_ids"].size())
        print(data["attention_mask"].size())
        print(data["label_groups"])
        spans = data["event_spans"][0]
        for j, sp_list in enumerate(spans):
            for sp in sp_list:
                print(tokenizer.decode(data["input_ids"][j][sp[0]:sp[1]]))
        break
# %%
# import json
# file = "../../data/processed/ACE/dev.jsonl"
# with open(file)as f:
#     lines = f.readlines()
# for line in lines[:50]:
#     data = json.loads(line)
#     # print(data.keys())
#     print(data["events"])
#     # break
# %%
