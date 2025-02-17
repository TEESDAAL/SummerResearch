run=30
models=( 'simple_pred' )

for model in "${models[@]}"
do
    echo $model
    qsub -t 1-$run:1 cogp.sh $model
done
