
# export MLFLOW_TRACKING_URI=http://experiments.traduce.zc2.ibm.com:5000
# export MLFLOW_S3_ENDPOINT_URL=http://data.digital\-Pathology.zc2.ibm.com:9000

# export AWS_ACCESS_KEY_ID="hun@zurich.ibm.com"
# export AWS_SECRET_ACCESS_KEY="ferry.polio.soigne.boom"

# export MLFLOW_EXPERIMENT_NAME=hun_VorHoverNet_training
# mlflow experiments create --artifact-location s3://mlflow -n ${MLFLOW_EXPERIMENT_NAME}
export MLFLOW_TRACKING_URI=file:///work/contluty01/IBM/VorHoVerNet/mlruns

# set variables
VERSION=00
DATASET=CoNSeP
STAIN_NORM=False
SPLIT=test

# other variables
if [ $STAIN_NORM == 'False' ]
then
    SN=0
else
    SN=1
fi

for idx in 01 02 03 04 05
do
    export MLFLOW_EXPERIMENT_NAME=hun_VorHoverNet_training
    MODEL_NAME=m_ver${VERSION}_sn${SN}_${idx}
    MODEL_DIR=/work/contluty01/IBM/VorHoVerNet/${MODEL_NAME}/checkpoints

    mlflow run --no-conda \
    /home/contluty01/Project/IBM/ninepins/experiments/VorHoVerNet_training \
    -P model_name=${MODEL_NAME} \
    -P data_path=/home/${USER}/Project/IBM/ninepins/histocartography/image/VorHoVerNet \
    -P version=${VERSION} \
    -P iteration=0 \
    -P early_stop_patience=10 \
    -P output_root=/work/${USER}/IBM/VorHoVerNet \
    -P log_interval=300 \
    -P load_pretrained=T \
    -P stain_norm=${STAIN_NORM}

    python /home/${USER}/Project/IBM/ninepins/histocartography/image/VorHoVerNet/run_inference.py \
    --model_dir=${MODEL_DIR} \
    --user=${USER} \
    --version=${VERSION}

    export MLFLOW_EXPERIMENT_NAME=tye_post_processing

    mlflow run --no-conda /home/${USER}/Project/IBM/ninepins/experiments/VorHoVerNet_post_processing \
    -P split=${SPLIT} -P prefix=dot_refinement \
    -P ckpt-filename=${MODEL_NAME} -P dataset=${DATASET} \
    -P version=1 -P inference-path=/work/${USER}/IBM/VorHoVerNet/inference \
    -P distancemap-threshold=2.3 -P find-best=1

    # mlflow run --no-conda /home/${USER}/Project/IBM/ninepins/experiments/VorHoVerNet_post_processing \
    # -P split=${SPLIT} -P prefix=dot_refinement \
    # -P ckpt-filename=${MODEL_NAME} -P dataset=${DATASET} \
    # -P version=2 -P inference-path=/work/${USER}/IBM/VorHoVerNet/inference \
    # -P distancemap-threshold=2.3 -P find-best=1

    mlflow run --no-conda /home/${USER}/Project/IBM/ninepins/experiments/VorHoVerNet_post_processing \
    -P split=${SPLIT} -P prefix=dot_refinement \
    -P ckpt-filename=${MODEL_NAME} -P dataset=${DATASET} \
    -P version=5 -P inference-path=/work/${USER}/IBM/VorHoVerNet/inference \
    -P find-best=1
done

export MLFLOW_EXPERIMENT_NAME=hun_VorHoverNet_training

# for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
# do
#     mlflow run --no-conda \
#     /home/${USER}/Project/IBM/ninepins/experiments/VorHoVerNet_training \
#     -P model_name=m_mixed_PseuRate_${rate}_02 \
#     -P data_path=/home/${USER}/Project/IBM/ninepins/histocartography/image/VorHoVerNet \
#     -P version=0 \
#     -P iteration=0 \
#     -P early_stop_patience=10 \
#     -P output_root=/work/${USER}/IBM/VorHoVerNet \
#     -P log_interval=300 \
#     -P load_pretrained=T \
#     -P data_mix_rate=${rate}
# done
