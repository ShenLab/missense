
alg=DNN

python calc_rank_score.py --bg_path all_snv/${alg}.sample_HS_SNV.csv --input_path ${alg}/final_HS/pred/asd_HS.csv --alg_name ${alg}
python calc_rank_score.py --bg_path all_snv/${alg}.sample_HS_SNV.csv --input_path ${alg}/final_HS/pred/chd_HS.csv  --alg_name ${alg}
python calc_rank_score.py --bg_path all_snv/${alg}.sample_HS_SNV.csv --input_path ${alg}/final_HS/pred/control_HS.csv  --alg_name ${alg}

python calc_rank_score.py --bg_path all_snv/${alg}.sample_HS_SNV.csv --input_path ${alg}/final_HS/pred/cancer_HS.csv  --alg_name ${alg}
