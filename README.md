# Are LLMs Good Annotators for Discourse-level Event Relation Extraction?

## Overview

This repository contains the code and datasets used in the paper:

> **"Are LLMs Good Annotators for Discourse-level Event Relation Extraction?"**\
> *Authors: Kangda Wei, Aayush Gautam, Ruihong Huang*\
> *Published in: Findings of the Association for Computational Linguistics: EMNLP 2024*

We systematically evaluate the performance of Large Language Models (LLMs) in extracting event relations using the MAVEN-ERE dataset. The study examines various prompting techniques and assesses LLMs like GPT-3.5 and Llama-2 against a supervised baseline presented in [MAVEN-ERE](https://aclanthology.org/2022.emnlp-main.60/) by Wang et al. 2022.

## Introduction

Event Relation Extraction (ERE) is a challenging NLP task involving the identification of event relations such as:

- **Coreference**
- **Temporal** (Before, After, Overlap, etc.)
- **Causal** (Cause, Precondition)
- **Subevent**

This study evaluates LLMs in document-level ERE using multiple prompting techniques and supervised fine-tuning.

## Dataset

We use the **MAVEN-ERE** dataset from Wang et al. (2022), which contains:

- **4,480** Wikipedia documents
- **103,193** event coreference chains
- **1,216,217** temporal relations
- **57,992** causal relations
- **15,841** subevent relations

## Methods

We implement four prompting strategies:

1. **Bulk Prediction** - Querying for all relations at once.
2. **Iterative Prediction** - Querying sentence by sentence, augmenting predictions.
3. **Event Ranking** - Ranking event by event.
4. **Pairwise Prediction** - Querying all event-TIMEX pairs for relation classification.

Models Evaluated:

- **GPT-3.5-turbo-16k** (OpenAI API)
- **Llama-2-7b-chat-hf** (Hugging Face)

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/WeiKangda/LLM-ERE.git
cd LLM-ERE
pip install -r requirements.txt
```

## Usage

To run the supervised baseline, follow the README file in the corresponding directory. The code is the same as [MAVEN-ERE](https://aclanthology.org/2022.emnlp-main.60/)

To run LLMs evluation, follow the README file in the ```bash ./llm``` directory.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{wei-etal-2024-llms,
    title = "Are {LLM}s Good Annotators for Discourse-level Event Relation Extraction?",
    author = "Wei, Kangda  and
      Gautam, Aayush  and
      Huang, Ruihong",
    editor = "Al-Onaizan, Yaser  and
      Bansal, Mohit  and
      Chen, Yun-Nung",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2024",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-emnlp.1/",
    doi = "10.18653/v1/2024.findings-emnlp.1",
    pages = "1--19",
    abstract = "Large Language Models (LLMs) have demonstrated proficiency in a wide array of natural language processing tasks. However, its effectiveness over discourse-level event relation extraction (ERE) tasks remains unexplored. In this paper, we assess the effectiveness of LLMs in addressing discourse-level ERE tasks characterized by lengthy documents and intricate relations encompassing coreference, temporal, causal, and subevent types. Evaluation is conducted using an commercial model, GPT-3.5, and an open-source model, LLaMA-2. Our study reveals a notable underperformance of LLMs compared to the baseline established through supervised learning. Although Supervised Fine-Tuning (SFT) can improve LLMs performance, it does not scale well compared to the smaller supervised baseline model. Our quantitative and qualitative analysis shows that LLMs have several weaknesses when applied for extracting event relations, including a tendency to fabricate event mentions, and failures to capture transitivity rules among relations, detect long distance relations, or comprehend contexts with dense event mentions."
}
```

## Acknowledgments

We gratefully acknowledge support from National Science Foun- dation via the award IIS-1942918. Portions of this research were conducted with the advanced com- puting resources provided by Texas A&M High- Performance Research Computing. The Llama-2 model was accessed via Hugging Face, and GPT-3.5 was evaluated using OpenAI’s API.

---

For questions, please open an issue or contact kangda@tamu.edu.

