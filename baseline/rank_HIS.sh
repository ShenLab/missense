
alg=DNN

python calc_rank_score.py --bg_path all_snv/${alg}.sample_HIS_SNV.csv --input_path ${alg}/final_HIS/pred/asd_HIS.csv --alg_name ${alg}
python calc_rank_score.py --bg_path all_snv/${alg}.sample_HIS_SNV.csv --input_path ${alg}/final_HIS/pred/chd_HIS.csv  --alg_name ${alg}
python calc_rank_score.py --bg_path all_snv/${alg}.sample_HIS_SNV.csv --input_path ${alg}/final_HIS/pred/control_HIS.csv  --alg_name ${alg}

python calc_rank_score.py --bg_path all_snv/${alg}.sample_HIS_SNV.csv --input_path ${alg}/final_HIS/pred/cancer_HIS.csv  --alg_name ${alg}
