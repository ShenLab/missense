#job_dir=v3_HIS
job_dir=final_HIS
#rm -rf $job_dir
mkdir -p $job_dir
#python main.py --input_config input_HIS_config.json  --job_dir ${job_dir} --mode train > ${job_dir}/train.log
python main.py --input_config input_HIS_config.json  --job_dir ${job_dir} --mode test > ${job_dir}/test.log
