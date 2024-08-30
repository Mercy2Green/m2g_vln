# python /home/lg1/.conda/envs/conceptgraph/lib/python3.10/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py \
#     --input_dir /home/lg1/peteryu_workspace/model/LLaMA --model_size 7B --output_dir /home/lg1/peteryu_workspace/model/llama-7b

python3 -m llava.model.apply_delta \
    --base /home/lg1/peteryu_workspace/model/llama-7b \
    --target /home/lg1/peteryu_workspace/model/LLaVA-7B-v0 \
    --delta liuhaotian/LLaVA-7b-delta-v0