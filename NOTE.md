```

# 模型质量测评
nohup bash batch_eval_vllm_single.sh qwen2.5 127500 http://10.xxx.8.1:8600/v1/chat/completions /workspace/models/Qwen2.5-7B-Instruct /workspace/data/cmrc_mixup outputs_vllm_yarn_wafp16 > outputs_vllm_yarn_wafp16.log 2>&1  &



# 数据转换

output_dir="/Users/liguodong/work/data/cmrc_mixup_jsonl"
data_path="/Users/liguodong/work/data/cmrc_mixup"
url="http://10.xxx.8.1:8600/v1/chat/completions"
model_path="/Users/liguodong/model/Qwen-1_8B-Chat"
model_name="qwen2"
model_max_len=127500

python3 origin_data_convert_jsonl.py --url $url --model-path $model_path --model-name $model_name --model-max-len $model_max_len --data-path $data_path --output-dir $output_dir

```
