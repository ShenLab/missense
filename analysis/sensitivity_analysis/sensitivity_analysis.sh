
ASD_HIS_path='../data/case_control/asd.anno.rare.final.HIS.reformat.cnn.rank.csv'
ASD_HS_path='../data/case_control/asd.anno.rare.final.HS.reformat.cnn.rank.csv'

CHD_HIS_path='../data/case_control/chd_yale.anno.rare.final.HIS.reformat.cnn.rank.csv'
CHD_HS_path='../data/case_control/chd_yale.anno.rare.final.HS.reformat.cnn.rank.csv'

Control_HIS_path='../data/case_control/control_1911.anno.rare.final.HIS.reformat.cnn.rank.csv'
Control_HS_path='../data/case_control/control_1911.anno.rare.final.HS.reformat.cnn.rank.csv'

sample_HIS_path='../data/sample_HIS_SNV.csv'
sample_HS_path='../data/sample_HS_SNV.csv'


run() {
    config=${1}
    exp_dir=${2}
    block_num=${3}
    mkdir -p ${exp_dir}
    for gene_type in HIS HS
    do
        python ./trainer.py \
            --config ${config} \
            --base_dir ${exp_dir} \
            --gene_type ${gene_type}
        model_path=${exp_dir}/best_model_${gene_type}.h5
        python ./predictor.py \
            --input_path ../data/sample_${gene_type}_SNV.csv \
            --output_path ${exp_dir}/sample_${gene_type}_pred.csv \
            --gene_type ${gene_type} \
            --model_path ${model_path} 
        sample_path=${exp_dir}/sample_${gene_type}_pred.csv
        for data in ASD CHD Control
        do
            data_path=${data}_${gene_type}_path
            data_path=${!data_path}
            output_path=${exp_dir}/${data}_${gene_type}_pred.csv
            python ./predictor.py \
                --input_path ${data_path} \
                --output_path ${output_path} \
                --gene_type ${gene_type} \
                --model_path ${model_path}
            python calc_rank_score.py \
                --input_path ${output_path} \
                --sample_path ${sample_path}
        done
    done

    for data in ASD CHD Control
    do
        python ./combine_HIS_HS.py ${exp_dir}/${data}_HIS_pred.rank.csv \
            ${exp_dir}/${data}_HS_pred.rank.csv \
            ${exp_dir}/${data}_All_pred.rank.csv
    done
    input_CHD_path=${exp_dir}/CHD_All_pred.rank.csv
    input_ASD_path=${exp_dir}/ASD_All_pred.rank.csv
    input_Control_path=${exp_dir}/Control_All_pred.rank.csv
    figure_dir=${exp_dir}/
    echo R CMD BATCH --no-save --no-restore "'--args chd_path=\"${input_CHD_path}\" asd_path=\"${input_ASD_path}\" control_path=\"${input_Control_path}\" block_num=\"${block_num}\" output_dir=\"${figure_dir}\"'" plot_figure_s.R
}

#run config_residual_block_16.json  res/residual_block_16
run config_residual_block_8.json  res/residual_block_8 8
run config_residual_block_4.json  res/residual_block_4 4
