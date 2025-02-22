import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from preprocess import *
import sys
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_doc_number", default=10, type=int)
    parser.add_argument("--valid_doc_number", default=10, type=int)
    parser.add_argument("--test_doc_number", default=10, type=int)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--without_downsample", action="store_true")
    parser.add_argument("--relation", default="coreference", choices=["coreference","temporal", "causal", "subevent", "joint"])
    args = parser.parse_args()

    # Model from Hugging Face hub
    base_model = "meta-llama/Llama-2-7b-chat-hf"

    # New instruction dataset
    data_dir = "../data/MAVEN_ERE"

    # Fine-tuned model
    new_model = f"./saved_models/llama-2-7b-chat-maven-ere-{args.relation}-{args.train_doc_number}"
    if args.without_downsample:
        new_model += "-without_downsample"
    else:
        new_model += "-with_downsample"
    # Cache directory for model and dataset
    cache_dir = "/scratch/user/kangda"
    ################################################################################
    # QLoRA parameters
    ################################################################################

    # LoRA attention dimension
    lora_r = 64

    # Alpha parameter for LoRA scaling
    lora_alpha = 16

    # Dropout probability for LoRA layers
    lora_dropout = 0.1

    ################################################################################
    # bitsandbytes parameters
    ################################################################################

    # Activate 4-bit precision base model loading
    use_4bit = True

    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = "float16"

    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = "nf4"

    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = False

    ################################################################################
    # TrainingArguments parameters
    ################################################################################

    # Output directory where the model predictions and checkpoints will be stored
    #output_dir = "./results"
    output_dir = f"./output/llama2_fine_tuning/{args.relation}/{args.train_doc_number}"
    if args.without_downsample:
        output_dir += "_without_downsample"
    else:
        output_dir += "_with_downsample"
    # Number of training epochs
    num_train_epochs = args.epoch

    # Enable fp16/bf16 training (set bf16 to True with an A100)
    fp16 = False
    bf16 = False

    # Batch size per GPU for training
    per_device_train_batch_size = 4

    # Batch size per GPU for evaluation
    per_device_eval_batch_size = 4

    # Number of update steps to accumulate the gradients for
    gradient_accumulation_steps = 1

    # Enable gradient checkpointing
    gradient_checkpointing = True

    # Maximum gradient normal (gradient clipping)
    max_grad_norm = 0.3

    # Initial learning rate (AdamW optimizer)
    learning_rate = 2e-4

    # Weight decay to apply to all layers except bias/LayerNorm weights
    weight_decay = 0.001

    # Optimizer to use
    optim = "paged_adamw_32bit"

    # Learning rate schedule (constant a bit better than cosine)
    lr_scheduler_type = "constant"

    # Number of training steps (overrides num_train_epochs)
    max_steps = -1

    # Ratio of steps for a linear warmup (from 0 to learning rate)
    warmup_ratio = 0.03

    # Group sequences into batches with same length
    # Saves memory and speeds up training considerably
    group_by_length = True

    # Save checkpoint every X updates steps
    save_steps = 2000

    # Log every X updates steps
    logging_steps = 1000

    ################################################################################
    # SFT parameters
    ################################################################################

    # Maximum sequence length to use
    max_seq_length = None

    # Pack multiple short examples in the same input sequence to increase efficiency
    packing = False

    # Load the entire model on the GPU 0
    device_map = {"": 0}

    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    if os.path.exists(os.path.join(output_dir, "log.txt")):
        os.remove(os.path.join(output_dir, "log.txt"))
    sys.stdout = open(os.path.join(output_dir, "log.txt"), 'w')
    # Load dataset
    train_docs = train_or_valid_doc_process(data_dir, split="train")[:args.train_doc_number]
    valid_docs = train_or_valid_doc_process(data_dir, split="valid")[:args.valid_doc_number]
    test_docs = test_doc_process(data_dir)[:args.test_doc_number]
    
    if args.without_downsample:
        coreference_none_drop_rate = 0
        temporal_none_drop_rate = 0
        causal_none_drop_rate = 0
        subevent_none_drop_rate = 0
        before_drop_rate = 0
    else:
        coreference_none_drop_rate = 0.95
        temporal_none_drop_rate = 0.90
        causal_none_drop_rate = 0.98
        subevent_none_drop_rate = 0.99
        before_drop_rate = 0.90

    if args.without_downsample:
        downsample = "without_downsample"
        train_file_path = os.path.join(data_dir, f"train_{args.relation}_{args.train_doc_number}_processed_without_downsample.jsonl")
        valid_file_path = os.path.join(data_dir, f"valid_{args.relation}_{args.valid_doc_number}_processed_without_downsample.jsonl")
        test_file_path = os.path.join(data_dir, f"test_{args.relation}_{args.test_doc_number}_processed_without_downsample.jsonl")
    else:
        downsample = "with_downsample"
        train_file_path = os.path.join(data_dir, f"train_{args.relation}_{args.train_doc_number}_processed_with_downsample.jsonl")
        valid_file_path = os.path.join(data_dir, f"valid_{args.relation}_{args.valid_doc_number}_processed_with_downsample.jsonl")
        test_file_path = os.path.join(data_dir, f"test_{args.relation}_{args.test_doc_number}_processed_with_downsample.jsonl")
    
    if not os.path.exists(train_file_path):
        processed_train_data = docs_to_pairs(train_docs, split="train", down_sample=downsample, relation=args.relation, doc_number=args.train_doc_number, coreference_none_drop_rate=coreference_none_drop_rate, temporal_none_drop_rate=temporal_none_drop_rate, causal_none_drop_rate=causal_none_drop_rate, subevent_none_drop_rate=subevent_none_drop_rate, before_drop_rate=before_drop_rate)
    if not os.path.exists(valid_file_path):
        processed_valid_data = docs_to_pairs(valid_docs, split="valid", down_sample=downsample, relation=args.relation, doc_number=args.valid_doc_number, coreference_none_drop_rate=coreference_none_drop_rate, temporal_none_drop_rate=temporal_none_drop_rate, causal_none_drop_rate=causal_none_drop_rate, subevent_none_drop_rate=subevent_none_drop_rate, before_drop_rate=before_drop_rate)
    if not os.path.exists(test_file_path):
        processed_test_data = docs_to_pairs(test_docs, split="test", down_sample=downsample, relation=args.relation, doc_number=args.test_doc_number, coreference_none_drop_rate=coreference_none_drop_rate, temporal_none_drop_rate=temporal_none_drop_rate, causal_none_drop_rate=causal_none_drop_rate, subevent_none_drop_rate=subevent_none_drop_rate, before_drop_rate=before_drop_rate)

    data_files = {"train": train_file_path, \
                  "validation": valid_file_path, \
                  "test": test_file_path}
    dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)
    # exit()

    train_context_dict = open_context_file(os.path.join(data_dir, f"train_{args.relation}_{args.train_doc_number}_context_dict_{downsample}.jsonl"))
    valid_context_dict = open_context_file(os.path.join(data_dir, f"valid_{args.relation}_{args.valid_doc_number}_context_dict_{downsample}.jsonl"))
    test_context_dict = open_context_file(os.path.join(data_dir, f"test_{args.relation}_{args.test_doc_number}_context_dict_{downsample}.jsonl"))

    train_dataset = dataset["train"].map(transform_train_conversation, fn_kwargs={"context_dict":train_context_dict})
    valid_dataset = dataset["validation"].map(transform_valid_test_conversation, fn_kwargs={"context_dict":valid_context_dict})
    test_dataset = dataset["test"].map(transform_valid_test_conversation, fn_kwargs={"context_dict":test_context_dict})
    # Load tokenizer and model with QLoRA configuration
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print("Your GPU supports bfloat16: accelerate training with bf16=True")
            print("=" * 80)

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        cache_dir=cache_dir
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Load LLaMA tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="tensorboard"
    )

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    # Train model
    if not args.test_only:
        trainer.train()

        # Save trained model
        trainer.model.save_pretrained(new_model)
        trainer.tokenizer.save_pretrained(new_model)

    # Evaluation
    tokenizer = AutoTokenizer.from_pretrained(new_model, trust_remote_code=True)
    ft_model = PeftModel.from_pretrained(model, os.path.join("./", new_model))
    ft_model.bfloat16()
    model.eval()

    result = {"id": None, "coreference": [], \
              "temporal_relations": {"BEFORE": [], "CONTAINS": [], "OVERLAP": [], "BEGINS-ON": [], "ENDS-ON": [], "SIMULTANEOUS": []}, \
              "causal_relations": {"CAUSE": [], "PRECONDITION": []}, \
              "subevent_relations": []}

    for i, data_point in enumerate(tqdm(valid_dataset)):
        if data_point["doc_id"] != result["id"] and i != 0:
            with open(os.path.join(output_dir, f"valid_{args.relation}_prediction.jsonl"), "a")as f:
                f.write(json.dumps(result))
                f.write("\n")
            result = {"id": data_point["doc_id"], "coreference": [], \
                      "temporal_relations": {"BEFORE": [], "CONTAINS": [], "OVERLAP": [], "BEGINS-ON": [], "ENDS-ON": [], "SIMULTANEOUS": []}, \
                      "causal_relations": {"CAUSE": [], "PRECONDITION": []}, \
                      "subevent_relations": []}
        #if i == 20: exit()
        model_input = tokenizer(data_point["text"], return_tensors="pt").to("cuda")
        with torch.no_grad():
            pred = tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=8)[0])
            pred = pred.split("[/INST]")[1]
            pred = pred.split(" ")
            #print(pred)

        if "COREFERENCE" in pred:
            in_cluster = False
            for k, cluster in enumerate(result["coreference"]):
                if data_point["event_1"] in cluster:
                    result["coreference"][k].append(data_point["event_2"])
                    in_cluster = True
                elif data_point["event_2"] in cluster:
                    result["coreference"][k].append(data_point["event_1"])
                    in_cluster = True
            if not in_cluster:
                result["coreference"].append([data_point["event_1"], data_point["event_2"]])
        elif "BEFORE" in pred:
            result["temporal_relations"]["BEFORE"].append([data_point["event_1"], data_point["event_2"]])
        elif "CONTAINS" in pred:
            result["temporal_relations"]["CONTAINS"].append([data_point["event_1"], data_point["event_2"]])
        elif "OVERLAP" in pred:
            result["temporal_relations"]["OVERLAP"].append([data_point["event_1"], data_point["event_2"]])
        elif "BEGINS-ON" in pred:
            result["temporal_relations"]["BEGINS-ON"].append([data_point["event_1"], data_point["event_2"]])
        elif "ENDS-ON" in pred:
            result["temporal_relations"]["ENDS-ON"].append([data_point["event_1"], data_point["event_2"]])
        elif "SIMULTANEOUS" in pred:
            result["temporal_relations"]["SIMULTANEOUS"].append([data_point["event_1"], data_point["event_2"]])
        elif "CAUSE" in pred:
            result["causal_relations"]["CAUSE"].append([data_point["event_1"], data_point["event_2"]])
        elif "PRECONDITION" in pred:
            result["causal_relations"]["PRECONDITION"].append([data_point["event_1"], data_point["event_2"]])
        elif "SUBEVENT" in pred:
            result["subevent_relations"].append([data_point["event_1"], data_point["event_2"]])
        if i == len(valid_dataset) - 1:
            with open(os.path.join(output_dir, f"valid_{args.relation}_prediction.jsonl"), "a")as f:
                f.write(json.dumps(result))
                f.write("\n")