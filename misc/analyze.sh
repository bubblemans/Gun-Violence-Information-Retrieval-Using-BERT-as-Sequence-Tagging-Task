for target_type in 'shooter' 'victim'
do
    for input_file in 'train.tsv' 'dev.tsv' 'test.tsv'
    do
        python analyze.py --input_file "data/${input_file}" --target_type $target_type
        echo '\n'
    done
done