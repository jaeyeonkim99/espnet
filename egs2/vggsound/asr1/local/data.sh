#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0


stage=1
stop_stage=100000
log "$0 $*"
. utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh

FOLD=${1:-1}
DATA_PREP_ROOT=${2:-"."}

if [ -z "${VGGSOUND}" ]; then
    log "Fill the value of 'VGGSOUND' of db.sh"
    exit 1
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Checking VGGSound data directory"
    
    # Check if VGGSOUND directory exists
    if [ ! -d "${VGGSOUND}" ]; then
        log "Error: ${VGGSOUND} directory does not exist!"
        exit 1
    fi
    
    # Count .mp4 files
    mp4_count=$(find "${VGGSOUND}" -name "*.mp4" -type f | wc -l)
    log "Found ${mp4_count} .mp4 files in ${VGGSOUND}"
    
    if [ ${mp4_count} -eq 0 ]; then
        log "Warning: No .mp4 files found in ${VGGSOUND}"
        log "Please download VGGSound videos first"
    else
        log "VGGSound data check completed successfully"
    fi
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Data Preparation"

    mkdir -p data/{train,valid,test}
    python3 local/data_prep.py ${VGGSOUND}
    for x in test valid train; do
        for f in text wav.scp utt2spk; do
            sort data/${x}/${f} -o data/${x}/${f}
        done
        utils/utt2spk_to_spk2utt.pl data/${x}/utt2spk > "data/${x}/spk2utt"
        utils/validate_data_dir.sh --no-feats data/${x} || exit 1
    done
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
