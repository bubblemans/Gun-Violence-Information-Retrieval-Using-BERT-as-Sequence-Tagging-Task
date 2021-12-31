for model_type in "Linear" "LSTM" "BiLSTM"
do
	for lr in 1e-5 5e-5 5e-6 
	do
		for batch_size in 2 4 8
		do
			for max_seq_length in 256
			do
				echo "$model_type $lr $epochs $batch_size $max_seq_length"
				python train.py --model_type $model_type --lr $lr --epochs 50 --batch_size $batch_size --max_seq_length $max_seq_length --input_dir shooter --output_dir shooter/output
			done
		done
	done
done
