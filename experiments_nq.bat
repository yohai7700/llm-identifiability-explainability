echo "Starting Experiments"
python main.py --task evaluate --source_dataset_type natural_questions_clean --training_llm_generating_model_name Qwen/Qwen2-0.5B-Instruct --eval_dataset_type yelp --llm_generating_model_name Qwen/Qwen2-0.5B-Instruct > ./experiments_results/yelp_qwen.txt
python main.py --task evaluate --source_dataset_type natural_questions_clean --training_llm_generating_model_name Qwen/Qwen2-0.5B-Instruct --eval_dataset_type yelp --llm_generating_model_name microsoft/Phi-3.5-mini-instruct > ./experiments_results/yelp_phi.txt
python main.py --task evaluate --source_dataset_type natural_questions_clean --training_llm_generating_model_name Qwen/Qwen2-0.5B-Instruct --eval_dataset_type amazon_polarity --llm_generating_model_name Qwen/Qwen2-0.5B-Instruct > ./experiments_results/amazon_polarity_qwen.txt
python main.py --task evaluate --source_dataset_type natural_questions_clean --training_llm_generating_model_name Qwen/Qwen2-0.5B-Instruct --eval_dataset_type amazon_polarity --llm_generating_model_name microsoft/Phi-3.5-mini-instruct > ./experiments_results/amazon_polarity_phi.txt
python main.py --task evaluate --source_dataset_type natural_questions_clean --training_llm_generating_model_name Qwen/Qwen2-0.5B-Instruct --eval_dataset_type natural_questions_clean --llm_generating_model_name Qwen/Qwen2-0.5B-Instruct > ./experiments_results/nqclean_qwen.txt
python main.py --task evaluate --source_dataset_type natural_questions_clean --training_llm_generating_model_name Qwen/Qwen2-0.5B-Instruct --eval_dataset_type natural_questions_clean --llm_generating_model_name microsoft/Phi-3.5-mini-instruct > ./experiments_results/nqclean_phi.txt


