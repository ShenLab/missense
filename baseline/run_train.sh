
gene_type=HS
model_name=DNN

job_dir=${model_name}/final_${gene_type}/
mkdir -p $job_dir
python main.py --input_config input_${gene_type}_config.json  --model_name ${model_name} --job_dir ${job_dir} --mode train #> ${job_dir}/train.log
python main.py --input_config input_${gene_type}_config.json  --model_name ${model_name} --job_dir ${job_dir} --mode test #> ${job_dir}/test.log
