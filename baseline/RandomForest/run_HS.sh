job_dir=v3_HS
job_dir=final_HS
#rm -rf $job_dir
mkdir -p $job_dir
#python main.py --input_config input_HS_config.json  --job_dir ${job_dir} --mode train > ${job_dir}/train.log
python main.py --input_config input_HS_config.json  --job_dir ${job_dir} --mode test > ${job_dir}/test.log
