
ROOT="/local-scratch/nishant/simul/speech_simul"
FAIRSEQ="${ROOT}/itst"
OG_FAIRSEQ="${ROOT}/fairseq"

mustc_root="/cs/natlang-expts/nishant/speech_to_text_data"
lang=de


# unzip
# tar -xzvf ${mustc_root}/MUSTC_v1.0_en-${lang}.tar.gz

# prepare ASR data
# python3 $OG_FAIRSEQ/examples/speech_to_text/prep_mustc_data_fast.py \
#   --data-root ${mustc_root} --task asr \
#   --vocab-type unigram --vocab-size 10000 \
#   --cmvn-type global \
#   --fbank-gcmvn-exist

# prepare ST data
python $OG_FAIRSEQ/examples/speech_to_text/prep_mustc_data_fast.py \
  --data-root ${mustc_root} --task st \
  --vocab-type unigram --vocab-size 10000 \
  --cmvn-type global \
  --fbank-gcmvn-exist

# generate the wav list and reference file for SimulEval
# eval_data="${mustc_root}/en-de-eval"
# mkdir -p $eval_data

# for split in dev tst-COMMON tst-HE
# do
#     python $FAIRSEQ/examples/speech_to_text/seg_mustc_data.py \
#     --data-root ${mustc_root} --lang ${lang} \
#     --split ${split} --task st \
#     --output ${eval_data}/${split}
# done
