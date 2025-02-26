## Usage
### Running GPT Models
To use the bulk prediction prompt for GPT models, run:
```bash
python gpt_bulk_prediction.py
```
To run other prompt types, change the Python file name to the corresponding prompt.

### Running LLaMA-2
To use bulk prediction with LLaMA-2, run:
```bash
torchrun --nproc_per_node 1 llama2_bulk_prediction.py \  
    --ckpt_dir ../../llama/llama-2-7b-chat/ \  
    --tokenizer_path ../../llama/tokenizer.model \  
    --max_seq_len 4096 --max_gen_len 200 --max_batch_size 6 --test True
```
To run other prompt types, change the Python file name to the corresponding prompt.

### Fine-tuning LLaMA-2
To fine-tune LLaMA-2, run:
```bash
python -u llama2_fine_tuning.py --relation joint --epoch 3 --train_doc_number 5 --without_downsample
```

## Post-processing
Before evaluating results, run the following script for post-processing:
```bash
python postprocess_bulk_prediction.py
```
For bulk prediction, ensure that you change the `output_dir` to the corresponding directory for the model you want to evaluate. The same applies to other prompting methods—change the script name accordingly.

## Evaluation
To obtain the final evaluation results, run:
```bash
python evaluate.py
```
Before running the evaluation, ensure that you change the `input_dir` and `output_dir` to match the correct directories for the model you want to evaluate.




