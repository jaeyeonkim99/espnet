#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

asr_speech_fold_length=1000 # 6.25 sec, because audio is 5 sec each.
inference_model=valid.acc.best.pth

n_folds=5 # This runs all 5 folds in parallel, take care.

asr_config=conf/beats_classification.yaml

mynametag=fast

train_set="train"
valid_set="val"
test_set="test"

./asr.sh \
    --asr_tag "${mynametag}" \
    --ngpu 1 \
    --stage 1 \
    --stop_stage 5 \
    --inference_args "--ctc_weight 0.0 --maxlenratio -1" \
    --token_type word \
    --asr_speech_fold_length ${asr_speech_fold_length} \
    --use_lm false \
    --feats_type raw \
    --max_wav_duration 6 \
    --feats_normalize utterance_mvn\
    --inference_nj 8 \
    --inference_asr_model ${inference_model} \
    --asr_config "${asr_config}" \
    --train_set "${train_set}" \
        --valid_set "${valid_set}" \
        --test_sets "${test_set}" "$@" &
