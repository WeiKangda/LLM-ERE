
torchrun --nproc_per_node 1 llama2_pairwise.py \
    --ckpt_dir ../../llama/llama-2-7b-chat/ \
    --tokenizer_path ../../llama/tokenizer.model \
    --max_seq_len 4096 --max_gen_len 200 --max_batch_size 6 --test False