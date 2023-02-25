
ROOT="/home/nishant/simul/speech_simul"
FAIRSEQ="${ROOT}/itst"
# OG_FAIRSEQ="${ROOT}/fairseq"

EXP="${ROOT}/experiments"
mkdir -p $EXP

mustc_root="${ROOT}/data/speech_to_text_data"
lang=de


pretrain_asr(){

    # conda activate speech_fx
    # echo "switched to conda env [speech_fx] .. "
    # sleep 5

    name="en_de"

    asr_modelfile="$EXP/asr/$name/checkpoints"
    logs="$EXP/asr/$name/logs"

    mkdir -p $logs $asr_modelfile

    python $OG_FAIRSEQ/train.py ${mustc_root}/en-${lang} \
    --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
    --save-dir ${asr_modelfile} \
    --num-workers 6 \
    --max-update 100000 \
    --max-tokens 30000  \
    --task speech_to_text \
    --criterion label_smoothed_cross_entropy --report-accuracy \
    --arch convtransformer_espnet \
    --optimizer adam \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --warmup-updates 10000 \
    --clip-norm 10.0 \
    --seed 1 \
    --update-freq 4 \
    --save-interval-updates 1000 \
    --keep-interval-updates 5 \
    --keep-last-epochs 5 \
    --find-unused-parameters \
    --fp16 \
    --log-format simple \
    --log-interval 100 \
    --wandb-project "mustc-en-de-asr" \
    | tee -a $logs/train_new_data.log
}

eval_asr(){

    # conda activate speech
    # echo "switched to conda env [speech] .. "
    # sleep 5

    # asr_modelfile="$EXP/asr/en_de/checkpoints/must_c_v1_en_de_pretrained_asr.pt"
    asr_modelfile="$EXP/asr/en_de/checkpoints/checkpoint_best.pt"
    mkdir -p "$EXP/asr/en_de/results/"

    python $OG_FAIRSEQ/fairseq_cli/generate.py ${mustc_root}/en-${lang} --config-yaml config_asr.yaml --gen-subset dev_asr \
        --task speech_to_text --path ${asr_modelfile} --max-tokens 50000 --max-source-positions 6000 --beam 1 \
        --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct \
        > "$EXP/asr/en_de/results/pred.out"

}



waitk_st(){
    # lagging number in wait-k policy
    k=$1
    name="wait_$k"

    # asr_modelfile="$EXP/asr/en_de/checkpoints/must_c_v1_en_de_pretrained_asr.pt"
    # asr_modelfile="$EXP/asr/en_de/checkpoints/checkpoint_last.pt"
    asr_modelfile="/cs/natlang-expts/nishant/speech_to_text_data/en-de/pretrained_simul_en_de/convtransformer_wait5_pre7.pt"
    
    st_modelfile="$EXP/st/$name/checkpoints"
    logs="$EXP/st/$name/logs"
    mkdir -p $logs $st_modelfile


    python $FAIRSEQ/train.py ${mustc_root}/en-${lang} \
        --config-yaml config_st_pre.yaml --train-subset train_st --valid-subset dev_st \
        --user-dir examples/simultaneous_translation \
        --save-dir ${st_modelfile} --num-workers 8  \
        --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
        --criterion label_smoothed_cross_entropy \
        --warmup-updates 4000 --max-update 300000 --max-tokens 28000 --seed 2 \
        --label-smoothing 0.1 \
        --task speech_to_text  \
        --arch convtransformer_simul_trans_espnet  \
        --simul-type waitk_fixed_pre_decision  \
        --waitk-lagging ${k} \
        --fixed-pre-decision-ratio 7 \
        --update-freq 5 \
        --save-interval-updates 1000 \
        --keep-interval-updates 10 \
        --find-unused-parameters \
        --fp16 \
        --log-format simple \
        --log-interval 100 \
        --wandb-project "mustc-en-de-simul" \
        --finetune-from-model "${asr_modelfile}" \
        | tee -a $logs/train_new_data.log

        # --load-pretrained-encoder-from ${asr_modelfile} \

}

moe_waitk_st_phase1(){
    # lagging number in wait-k policy
    name="moe_waitk"

    waitk_model_for_encoder="$EXP/st/wait_5/checkpoint_last.pt"
    
    st_modelfile="$EXP/st/$name/checkpoints"
    logs="$EXP/st/$name/logs"
    mkdir -p $logs $st_modelfile

    python $FAIRSEQ/train.py ${mustc_root}/en-${lang} \
        --config-yaml config_st_pre.yaml --train-subset train_st --valid-subset dev_st \
        --user-dir examples/simultaneous_translation \
        --save-dir ${st_modelfile} --num-workers 8  \
        --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
        --criterion label_smoothed_cross_entropy \
        --warmup-updates 4000 --max-update 6000 --max-tokens 28000 --seed 2 \
        --label-smoothing 0.1 \
        --task speech_to_text  \
        --arch convtransformer_simul_trans_espnet  \
        --simul-type MoE_waitk_fixed_pre_decision  \
        --multipath \
        --load-pretrained-encoder-from ${waitk_model_for_encoder} \
        --freeze-pretrained-encoder \
        --fixed-pre-decision-ratio 7 \
        --update-freq 5 \
        --save-interval-updates 1000 \
        --keep-interval-updates 10 \
        --keep-last-epochs 10 \
        --find-unused-parameters \
        --fp16 \
        --log-format simple \
        --log-interval 100 \
        --wandb-project "mustc-en-de-simul" \
        | tee -a $logs/phase1_train.log

        # --load-pretrained-encoder-from ${asr_modelfile} \
}

moe_waitk_st_phase2(){
    # simply continue training where phase 1 finish
    # remove explicit encoder loading etc
    # NOTE: aim for net bsz 28000 x 5 x 2

    name="moe_waitk"

    moe_waitk_st_phase1_model="$EXP/st/moe_waitk/checkpoint_last.pt"
    
    st_modelfile="$EXP/st/$name/checkpoints"
    logs="$EXP/st/$name/logs"
    mkdir -p $logs $st_modelfile

    python $FAIRSEQ/train.py ${mustc_root}/en-${lang} \
        --config-yaml config_st_pre.yaml --train-subset train_st --valid-subset dev_st \
        --user-dir examples/simultaneous_translation \
        --save-dir ${st_modelfile} --num-workers 4  \
        --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
        --criterion label_smoothed_cross_entropy \
        --warmup-updates 4000 --max-update 25000 --max-tokens 28000 --seed 2 \
        --label-smoothing 0.1 \
        --task speech_to_text  \
        --arch convtransformer_simul_trans_espnet  \
        --simul-type MoE_waitk_fixed_pre_decision  \
        --multipath \
        --fixed-pre-decision-ratio 7 \
        --reset-parameter-state \
        --update-freq 10 \
        --save-interval-updates 1000 \
        --keep-interval-updates 10 \
        --keep-last-epochs 10 \
        --find-unused-parameters \
        --fp16 \
        --log-format simple \
        --log-interval 100 \
        --load-pretrained-encoder-from "${moe_waitk_st_phase1_model}" \
        --load-pretrained-decoder-from "${moe_waitk_st_phase1_model}" \
        --reset-dataloader \
        --reset-lr-scheduler \
        --reset-meters \
        --reset-optimizer \
        --wandb-project "mustc-en-de-simul" \
        | tee -a $logs/phase2_train.log
        # --load-pretrained-encoder-from ${asr_modelfile} \
        # --wandb-project "mustc-en-de-simul" \
}

itst(){

    name="itst"

    fb_modelfile="/cs/natlang-expts/nishant/speech_to_text_data/en-de/pretrained_simul_en_de/convtransformer_wait5_pre7.pt"
    
    st_modelfile="$EXP/st/$name/checkpoints"
    logs="$EXP/st/$name/logs"
    mkdir -p $logs $st_modelfile

    python $FAIRSEQ/train.py "${mustc_root}/en-de" \
        --config-yaml config_st_pre.yaml --train-subset train_st --valid-subset dev_st \
        --user-dir examples/simultaneous_translation \
        --save-dir ${st_modelfile} --num-workers 4  \
        --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
        --criterion label_smoothed_cross_entropy_with_itst_s2t_fixed_predecision \
        --warmup-updates 4000 --max-update 40000 --max-tokens 28000 --seed 2 \
        --label-smoothing 0.1 \
        --task speech_to_text  \
        --arch convtransformer_simul_trans_itst_espnet  \
        --unidirectional-encoder \
        --simul-type ITST_fixed_pre_decision  \
        --fixed-pre-decision-ratio 7 \
        --finetune-from-model ${fb_modelfile} \
        --update-freq 10 \
        --save-interval-updates 1000 \
        --keep-interval-updates 5 \
        --keep-last-epochs 10 \
        --find-unused-parameters \
        --fp16 \
        --log-format simple \
        --log-interval 100 \
        --wandb-project "mustc-en-de-simul" \
        | tee -a $logs/train.log
        
}

itst_long(){

    name="itst_long"

    # fb_modelfile="/cs/natlang-expts/nishant/speech_to_text_data/en-de/pretrained_simul_en_de/convtransformer_wait5_pre7.pt"
    waitk_model_for_encoder="$EXP/st/wait_5/checkpoint_last.pt"

    st_modelfile="$EXP/st/$name/checkpoints"
    logs="$EXP/st/$name/logs"
    mkdir -p $logs $st_modelfile

    python $FAIRSEQ/train.py "${mustc_root}/en-de" \
        --config-yaml config_st_pre.yaml --train-subset train_cloud_st --valid-subset dev_cloud_st \
        --user-dir examples/simultaneous_translation \
        --save-dir ${st_modelfile} --num-workers 12  \
        --optimizer adam --lr 0.0004 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
        --criterion label_smoothed_cross_entropy_with_itst_s2t_fixed_predecision \
        --warmup-updates 4000 --max-update 100000 --max-tokens 35000 --seed 2 \
        --label-smoothing 0.1 \
        --task speech_to_text  \
        --arch convtransformer_simul_trans_itst_espnet  \
        --unidirectional-encoder \
        --simul-type ITST_fixed_pre_decision  \
        --load-pretrained-encoder-from "${waitk_model_for_encoder}" \
        --fixed-pre-decision-ratio 7 \
        --threshold-delta 0.5 \
        --threshold-denom 60000 \
        --update-freq 2 \
        --save-interval-updates 1000 \
        --keep-interval-updates 20 \
        --keep-last-epochs 20 \
        --fp16 \
        --log-format simple \
        --log-interval 100 \
        --wandb-project "mustc-en-de-simul" \
        --empty-cache-freq 120 \
        | tee -a $logs/train.log
        
}

itst_long_continue(){

    name="itst_long"

    # fb_modelfile="/cs/natlang-expts/nishant/speech_to_text_data/en-de/pretrained_simul_en_de/convtransformer_wait5_pre7.pt"
    waitk_model_for_encoder="$EXP/st/wait_5/checkpoint_last.pt"

    st_modelfile="$EXP/st/$name/checkpoints"
    logs="$EXP/st/$name/logs"
    mkdir -p $logs $st_modelfile

    python $FAIRSEQ/train.py "${mustc_root}/en-de" \
        --config-yaml config_st_pre.yaml --train-subset train_cloud_st --valid-subset dev_cloud_st \
        --user-dir examples/simultaneous_translation \
        --save-dir ${st_modelfile} --num-workers 12  \
        --optimizer adam --lr 0.0004 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
        --criterion label_smoothed_cross_entropy_with_itst_s2t_fixed_predecision \
        --warmup-updates 4000 --max-update 100000 --max-tokens 35000 --seed 2 \
        --label-smoothing 0.1 \
        --task speech_to_text  \
        --arch convtransformer_simul_trans_itst_espnet  \
        --unidirectional-encoder \
        --simul-type ITST_fixed_pre_decision  \
        --fixed-pre-decision-ratio 7 \
        --threshold-delta 0.5 \
        --threshold-denom 60000 \
        --update-freq 2 \
        --save-interval-updates 1500 \
        --keep-interval-updates 40 \
        --keep-last-epochs 20 \
        --keep-best-checkpoints 10 \
        --fp16 \
        --log-format simple \
        --log-interval 100 \
        --wandb-project "mustc-en-de-simul" \
        --empty-cache-freq 120 \
        | tee -a $logs/train.log
        
}

################################


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
# itst
itst_long_continue