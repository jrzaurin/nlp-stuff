python run_bert.py --model_name "distilbert-base-uncased" --freeze_bert --lr 0.001 --save_results
python run_bert.py --model_name "distilbert-base-uncased" --freeze_bert --head_hidden_dim "[256]" --lr 0.001 --save_results

python run_bert.py --model_name "distilbert-base-uncased" --batch_size 32 --lr 5e-5 --save_results
python run_bert.py --model_name "distilbert-base-uncased" --with_scheduler --batch_size 32 --lr 5e-5 --save_results
python run_bert.py --model_name "distilbert-base-uncased" --with_scheduler --batch_size 32 --lr 1e-4 --save_results
