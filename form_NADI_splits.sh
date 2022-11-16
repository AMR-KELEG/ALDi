mkdir -p data/NADI_mapped/

for SPLIT in "train" "dev"
do
    grep "Egypt" "data/NADI2021/NADI2021_DEV.1.0/Subtask_1.1+2.1_MSA/MSA_${SPLIT}_labeled.tsv" \
        | cut -f 2 > data/NADI_mapped/${SPLIT}_MSA.txt
    grep "Egypt" "data/NADI2021/NADI2021_DEV.1.0/Subtask_1.2+2.2_DA/DA_${SPLIT}_labeled.tsv" \
        | cut -f 2 > data/NADI_mapped/${SPLIT}_DA.txt
done

python prepare_NADI_test.py