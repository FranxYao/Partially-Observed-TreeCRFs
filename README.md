![title](doc/title.png)

Yao Fu, Chuanqi Tan, Mosha Chen, Songfang Huang and Fei Huang. _Nested Named Entity Recognition with Partially Observed TreeCRFs_. AAAI 2021. [[arxiv](https://arxiv.org/abs/2012.08478)]


Dependency:

* [torch-struct](https://github.com/harvardnlp/pytorch-struct)
* [transformers](https://github.com/huggingface/transformers)

Train:
```bash
python train.py --output_dir outputs --model_type bert --config_name bert-base-uncased --model_name_or_path bert-base-uncased --train_file data/train.data --predict_file data/dev.data --test_file data/test.data --max_seq_length 64 --per_gpu_train_batch_size 48 --per_gpu_eval_batch_size 48 --do_train --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 --dataset GENIA --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine --latent_size 1 --seed 12345
```

Test:
```bash
python train.py --output_dir {CHECKPOINT_DIR} --model_type bert --config_name {BERT_CONFIG} --model_name_or_path {BERT_DIR} --train_file {TRAIN_FILE} --predict_file {DEV_FILE} --test_file {TEST_FILE} --max_seq_length 128 --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 --do_predict --learning_rate 3e-5 --num_train_epochs 100 --overwrite_output_dir --save_steps 1000 --dataset {DATASET_NAME}} --potential_normalization True --structure_smoothing_p 0.98 --parser_type deepbiaffine --latent_size 1 --seed 12345
```
