@echo off
rem Function to process datasets
:process_datasets
setlocal
set "source_dataset_types=amazon_polarity natural_questions_clean yelp"
set "models=Qwen/Qwen2-0.5B-Instruct microsoft/Phi-3.5-mini-instruct"
set "evaluation_datasets=yelp amazon_polarity natural_questions_clean"

for %%s in (%source_dataset_types%) do (
    for %%e in (%evaluation_datasets%) do (
        echo "Running experiment with source_dataset_type=%%s, training_model=Qwen/Qwen2-0.5B-Instruct, evaluation_dataset=%%e, llm_model=qwen"
        echo "python main.py --task evaluate --source_dataset_type %%s --training_llm_generating_model_name %training_model% --eval_dataset_type %%e --llm_generating_model_name Qwen/Qwen2-0.5B-Instruct --include_training_subset_size_in_classifier_folder True --training_subset_size 2000 > ./experiments_results/yelp2000_qwen_results/%%e_qwen.txt"
        echo "./experiments_results/yelp2000_qwen_results/%%e_qwen.txt"
        python main.py --task evaluate --source_dataset_type %%s --training_llm_generating_model_name Qwen/Qwen2-0.5B-Instruct --eval_dataset_type %%e --llm_generating_model_name Qwen/Qwen2-0.5B-Instruct --include_training_subset_size_in_classifier_folder True --training_subset_size 2000 > ./experiments_results/yelp2000_qwen_results/%%e_qwen.txt

        echo "Running experiment with source_dataset_type=%%s, training_model=Qwen/Qwen2-0.5B-Instruct, evaluation_dataset=%%e, llm_model=phi"
        echo "./experiments_results/yelp2000_qwen_results/%%e_phi.txt"
        python main.py --task evaluate --source_dataset_type %%s --training_llm_generating_model_name Qwen/Qwen2-0.5B-Instruct --eval_dataset_type %%e --llm_generating_model_name microsoft/Phi-3.5-mini-instruct --include_training_subset_size_in_classifier_folder True --training_subset_size 2000 > ./experiments_results/yelp2000_qwen_results/%%e_phi.txt
    )
)
endlocal
exit /b

rem Main script
echo Starting Experiments
call :process_datasets