
ROOT="/home/nishant/simul/speech_simul"
FAIRSEQ="${ROOT}/itst"

EXP="${ROOT}/experiments/st"

mustc_root="${ROOT}/data/speech_to_text_data"
lang=de

eval_data="${mustc_root}/en-de-eval"

gen_itst_dev(){

    split="dev"
    name="itst"

    threshold=$1
    echo "Threshold: $threshold"

    expt="${EXP}/${name}"
    checkpoints="${expt}/checkpoints"

    results="${expt}/results/${threshold}/${split}"
    mkdir -p ${results}

    wav_list="${eval_data}/${split}/${split}.wav_list"
    reference="${eval_data}/${split}/${split}.de"

    ckpt="${checkpoints}/checkpoint_best.pt"

    # average last 5 checkpoints
    # python ${FAIRSEQ}/scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt --last_file ${last_file}
    # file=${modelfile}/average-model.pt 

    simuleval --agent ${FAIRSEQ}/examples/speech_to_text/simultaneous_translation/agents/simul_agent.s2t.itst.fixed_predecision.py \
        --source ${wav_list} \
        --target ${reference} \
        --data-bin ${mustc_root}/en-de \
        --config config_st_pre.yaml \
        --model-path ${ckpt} \
        --test-threshold ${threshold} \
        --output ${results} \
        --scores --gpu \
        --num-processes 1 \
        --port 12321 \
        | tee -a "${results}/out.log"

}


gen_itst_test(){

    split="tst-COMMON"
    name="itst_debug_load_encoder" #"itst"

    threshold=$1

    expt="${EXP}/${name}"
    checkpoints="${expt}/checkpoints"

    results="${expt}/results/${threshold}/${split}"
    mkdir -p ${results}

    wav_list="${eval_data}/${split}/${split}.wav_list"
    reference="${eval_data}/${split}/${split}.de"

    ckpt="${checkpoints}/checkpoint_best.pt"

    # average last 5 checkpoints
    # python ${FAIRSEQ}/scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt --last_file ${last_file}
    # file=${modelfile}/average-model.pt 

    simuleval --agent ${FAIRSEQ}/examples/speech_to_text/simultaneous_translation/agents/simul_agent.s2t.itst.fixed_predecision.py \
        --source ${wav_list} \
        --target ${reference} \
        --data-bin ${mustc_root}/en-de \
        --config config_st_pre.yaml \
        --model-path ${ckpt} \
        --test-threshold ${threshold} \
        --output ${results} \
        --scores --gpu \
        --num-processes 12 \
        --port 12321 \
        | tee -a "${results}/out.log"

}

export CUDA_VISIBLE_DEVICES=0

# gen_itst_dev [threshold]

# gen_itst_dev 0.2

gen_itst_test 0.5