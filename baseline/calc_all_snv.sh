
#python select_snv.py  --input_prefix /data/hq2130/large_files/rare_missense_id.anno.rare.HIS.reformat.csv  --output_path all_snv/sample_HIS_SNV.csv --ratio 0.037673058729887484

#python select_snv.py  --input_prefix /data/hq2130/large_files/rare_missense_id.anno.rare.HS.reformat.csv  --output_path all_snv/sample_HS_SNV.csv --ratio 0.022265192272898696


python prioritize_all.py --input_config all_HIS_SNV.json --model_dir DNN/final_HIS/model --input_prefix all_snv/sample_HIS_SNV --output_dir all_snv --model_name DNN

python prioritize_all.py --input_config all_HS_SNV.json --model_dir DNN/final_HS/model --input_prefix all_snv/sample_HS_SNV --output_dir all_snv --model_name DNN
