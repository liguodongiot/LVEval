model_name=$1
model_max_len=$2
url=$3
model_path=$4
data_path=$5
output_dir_prefix=$6

# timestamp=`date +"%Y%m%d-%H%M%S"`
timestamp=000000000000

output_dir="$output_dir_prefix/$model_name/$model_max_len/$timestamp"

echo "output dir $output_dir"

cmd="python3 prediction_vllm.py --url $url --model-path $model_path --model-name $model_name --model-max-len $model_max_len --data-path $data_path --output-dir $output_dir"
echo $cmd
eval $cmd

cmd="python3 evaluation.py --input-dir $output_dir"
echo $cmd
eval $cmd
