
# export MLFLOW_TRACKING_URI=http://experiments.traduce.zc2.ibm.com:5000
# export MLFLOW_S3_ENDPOINT_URL=http://data.digital\-Pathology.zc2.ibm.com:9000

# export AWS_ACCESS_KEY_ID="hun@zurich.ibm.com"
# export AWS_SECRET_ACCESS_KEY="ferry.polio.soigne.boom"

export MLFLOW_EXPERIMENT_NAME=hun_VorHoverNet_training
# mlflow experiments create --artifact-location s3://mlflow -n ${MLFLOW_EXPERIMENT_NAME}
export MLFLOW_TRACKING_URI=file:///work/contluty01/IBM/VorHoVerNet/mlruns

# mlflow.set_tracking_uri
mlflow run --no-conda /home/contluty01/Project/IBM/ninepins/experiments/VorHoVerNet_training \
-P model_name=model_test_01 \
-P batch_size=12 \
-P data_path=/home/contluty01/Project/IBM/ninepins/histocartography/image/VorHoVerNet \
-P number_of_workers=1 \
-P version=0 \
-P iteration=0 \
-P early_stop_patience=10 \
-P output_root=/work/contluty01/IBM/VorHoVerNet \
-P log_interval=300 \
-P load_pretrained=T
