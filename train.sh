
# export MLFLOW_TRACKING_URI=http://experiments.traduce.zc2.ibm.com:5000
# export MLFLOW_S3_ENDPOINT_URL=http://data.digital\-Pathology.zc2.ibm.com:9000

# export AWS_ACCESS_KEY_ID="hun@zurich.ibm.com"
# export AWS_SECRET_ACCESS_KEY="ferry.polio.soigne.boom"

export MLFLOW_EXPERIMENT_NAME=hun_VorHoverNet_training
# mlflow experiments create --artifact-location s3://mlflow -n ${MLFLOW_EXPERIMENT_NAME}
export MLFLOW_TRACKING_URI=file:///work/contluty01/IBM/VorHoVerNet/mlruns

# mlflow.set_tracking_uri

mlflow run --no-conda \
/home/contluty01/Project/IBM/ninepins/experiments/VorHoVerNet_training \
-P model_name=m_crfloss_01 \
-P data_path=/home/contluty01/Project/IBM/ninepins/histocartography/image/VorHoVerNet \
-P version=0 \
-P iteration=0 \
-P early_stop_patience=10 \
-P output_root=/work/contluty01/IBM/VorHoVerNet \
-P log_interval=300 \
-P load_pretrained=T
# -P batch_size=24 \
# -P number_of_workers=2 \

# for rate in 0.6 0.7 0.8 0.9 1.0
# do
#     mlflow run --no-conda \
#     /home/contluty01/Project/IBM/ninepins/experiments/VorHoVerNet_training \
#     -P model_name=m_mixed_PseuRate_${rate}_01 \
#     -P batch_size=12 \
#     -P data_path=/home/contluty01/Project/IBM/ninepins/histocartography/image/VorHoVerNet \
#     -P number_of_workers=1 \
#     -P version=0 \
#     -P iteration=0 \
#     -P early_stop_patience=10 \
#     -P output_root=/work/contluty01/IBM/VorHoVerNet \
#     -P log_interval=300 \
#     -P load_pretrained=T \
#     -P data_mix_rate=${rate}
# done
