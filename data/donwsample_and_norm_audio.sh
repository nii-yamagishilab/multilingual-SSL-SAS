source ./env.sh

downsamp_norm=True
if [ ${downsamp_norm} == 'True' ]; then

    cd $PWD
    ## downsample and normalization
    SOURCE_DIR=data/LibriSpeech/train-clean-100/
    TARGET_DIR=data/librispeech_100_wav16k_norm

    SAMP=16000

    ##########
    TMP=${TARGET_DIR}_TMP
    CONVERTED_TMP=${TARGET_DIR}_CONVERTED
    mkdir -p ${TMP}
    mkdir -p ${TARGET_DIR}
    mkdir -p ${CONVERTED_TMP}

    find $SOURCE_DIR -type f -name "*.flac" > file.lst
    # Step 1. Convert FLAC to WAV using ffmpeg
    cat file.lst | parallel -j 20 ffmpeg -i {} ${CONVERTED_TMP}/{/.}.wav
    # Clean up the list and re-generate with converted files
    find ${CONVERTED_TMP} -type f -name "*.wav" > file_converted.lst

    # Step 2. Down-sampling
    cat file_converted.lst | parallel -j 20 sh scripts/sub_down_sample.sh {} ${TMP}/{/.}.wav ${SAMP}
    #wait

    find ${TMP} -type f -name "*.wav" > file_tmp.lst
    # Step 3. Further processing
    cat file_tmp.lst | parallel -j 20 bash scripts/sub_sv56.sh {} ${TARGET_DIR}/{/.}.wav
    wait

    # Cleanup
    rm -r $TMP
    rm -r $CONVERTED_TMP
    rm file.lst
    rm file_converted.lst
    rm file_tmp.lst
fi
