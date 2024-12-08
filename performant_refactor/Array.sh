run=30
models=( 'simple_pred' 'complex_num_pred' 'complex_pred' 'MLGP' )

for model in "${models[@]}"
do
    echo $model
    qsub -t 1-$run:1 Job.sh $model
done
