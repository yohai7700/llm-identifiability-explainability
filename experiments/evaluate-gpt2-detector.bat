echo "Starting Experiments"
python main.py --task evaluate --source_dataset_type yelp --llm_generating_model_name Qwen/Qwen2-0.5B-Instruct > ./experiments_results/gpt2-detector/yelp_qwen.txt
python main.py --task evaluate --source_dataset_type amazon_polarity --llm_generating_model_name Qwen/Qwen2-0.5B-Instruct > ./experiments_results/gpt2-detector/amazon_polarity_qwen.txt
python main.py --task evaluate --source_dataset_type natural_questions_clean --llm_generating_model_name Qwen/Qwen2-0.5B-Instruct > ./experiments_results/gpt2-detector/natural_questions_qwen.txt
python main.py --task evaluate --source_dataset_type yelp --llm_generating_model_name microsoft/Phi-3.5-mini-instruct > ./experiments_results/gpt2-detector/yelp_phi.txt
python main.py --task evaluate --source_dataset_type amazon_polarity --llm_generating_model_name microsoft/Phi-3.5-mini-instruct > ./experiments_results/gpt2-detector/amazon_polarity_phi.txt
python main.py --task evaluate --source_dataset_type natural_questions_clean --llm_generating_model_name microsoft/Phi-3.5-mini-instruct > ./experiments_results/gpt2-detector/natural_questions_phi.txt