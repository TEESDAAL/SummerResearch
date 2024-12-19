run=30
models=( 'MLGP' 'simple_pred' 'complex_pred' 'complex_num_pred' )

for model in "${models[@]}"
do
    echo $model
    qsub -t 1-$run:1 Job.sh $model
done
