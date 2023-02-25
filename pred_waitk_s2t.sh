
ROOT="/local-scratch/nishant/simul/speech_simul"
FAIRSEQ="${ROOT}/itst"
OG_FAIRSEQ="${ROOT}/fairseq"

EXP="${ROOT}/experiments/st"

mustc_root="/cs/natlang-expts/nishant/speech_to_text_data"
lang=de

# modelfile=PATH_TO_SAVE_MODEL
# last_file=LAST_CHECKPOINT

eval_data="${mustc_root}/en-de-eval"

gen_waitk_dev(){

    split="dev"
    name="wait_5"

    expt="${EXP}/${name}"
    checkpoints="${expt}/checkpoints"

    results="${expt}/results/${split}"
    mkdir -p ${results}

    wav_list="${eval_data}/${split}/${split}.wav_list"
    reference="${eval_data}/${split}/${split}.de"

    ckpt="${checkpoints}/checkpoint_best.pt"

    # average last 5 checkpoints
    # python ${FAIRSEQ}/scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt --last_file ${last_file}
    # file=${modelfile}/average-model.pt 

    simuleval --agent ${FAIRSEQ}/examples/speech_to_text/simultaneous_translation/agents/fairseq_simul_st_agent.py \
        --source ${wav_list} \
        --target ${reference} \
        --data-bin ${mustc_root}/en-de \
        --config config_st_pre.yaml \
        --model-path ${ckpt} \
        --output ${results} \
        --scores --gpu \
        --num-processes 12 \
        --port 12321 \
        | tee -a "${results}/out.log"

}


gen_waitk_test(){

    split="tst-COMMON"
    name="wait_5"

    expt="${EXP}/${name}"
    checkpoints="${expt}/checkpoints"

    results="${expt}/results/${split}"
    mkdir -p ${results}

    wav_list="${eval_data}/${split}/${split}.wav_list"
    reference="${eval_data}/${split}/${split}.de"

    ckpt="${checkpoints}/checkpoint_best.pt"

    # average last 5 checkpoints
    # python ${FAIRSEQ}/scripts/average_checkpoints.py --inputs ${modelfile} --num-update-checkpoints 5 --output ${modelfile}/average-model.pt --last_file ${last_file}
    # file=${modelfile}/average-model.pt 

    simuleval --agent ${FAIRSEQ}/examples/speech_to_text/simultaneous_translation/agents/fairseq_simul_st_agent.py \
        --source ${wav_list} \
        --target ${reference} \
        --data-bin ${mustc_root}/en-de \
        --config config_st_pre.yaml \
        --model-path ${ckpt} \
        --output ${results} \
        --scores --gpu \
        --num-processes 12 \
        --port 1234 \
        | tee -a "${results}/out.log"

}

export CUDA_VISIBLE_DEVICES=0

gen_waitk_dev