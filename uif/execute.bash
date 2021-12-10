read -p "dataset (adult / bank / law): " dataset

coeff=(0.0 0.01 0.1 0.2 0.5 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 50.0 70.0 80.0 90.0 100.0 150.0 200.0 250.0 300.0 500.0 700.0 1000.0)
taus=(0.01 0.05 0.1 0.2)

if [ ${dataset} == "adult" ]
then
    device=1
fi

for seed in {2021..2023}
do
    # CUDA_VISIBLE_DEVICES=1,0 python3.6 main.py --dataset $dataset --lmdaF 0.0 --seed $seed --alg testfmean --fairtarget None
    for tau in "${taus[@]}"
    do
        for fair in "${coeff[@]}"
        do
            CUDA_VISIBLE_DEVICES=1,0 python3.6 main.py --dataset $dataset --lmda $fair --tau $tau --seed $seed --device $device
            CUDA_VISIBLE_DEVICES=1,0 python3.6 main.py --dataset $dataset --lmda $fair --tau $tau --seed $seed --mode hinge --device $device
        done
    done
done