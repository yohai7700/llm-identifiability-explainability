
import torch
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers_interpret import SequenceClassificationExplainer

from training.trainer import get_classification_model_folder
from data.utils.preprocessing import get_preprocessed_dataset_path

from data.list_dataset import ListDataset
from args import get_args


def get_explainer():
    model_folder = f'{get_classification_model_folder()}/model'
    model = AutoModelForSequenceClassification.from_pretrained(model_folder, device_map='cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_folder, device_map='cuda')
    return SequenceClassificationExplainer(
        model,
        tokenizer,
        custom_labels=["Human-Authentic", "LLM-Generated"]
    )

def interpret():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    eval_dataset = ListDataset(torch.load(get_preprocessed_dataset_path('eval'), weights_only=True))
    cls_explainer = get_explainer()
    if get_args().index is not None:
        try:
            index = get_args().index
            text = eval_dataset[index]['text']
            attributions = cls_explainer(text)
            cls_explainer.visualize(f'./interpreter_visualizations/vis-{index}.html')
            print(' '.join(map(lambda x: f'\score{{{x[1]}}}{{{x[0]}}}', attributions)))
        except Exception as exception:
            print(f'Error interpreting index {index}: {exception}')
    else:
        for index in tqdm(range(50)):
            try:
                text = eval_dataset[index]['text']
                cls_explainer(text)
                cls_explainer.visualize(f'./interpreter_visualizations/vis-{index}.html')
            except Exception as exception:
                print(f'Error interpreting index {index}: {exception}')

def interpret_count_candidates():
    CANDIDATES = ['@', 'rot', 'sourced', 'dev', 'stole', 'describes', 'mainly', '##hai', 'border', 'wo', 'combines', 'oct', 'harp', 'washington', 'proceed', 'ku', 'grind', 'corp', 'navy', 'dramatically', 'resources', 'conduct', 'additionally', '2009', '##list', 'session', 'selected', '=', 'character', 'embarked', 'model', 'bain', 'angeles', 'featuring', 'korean', 'tampa', 'contains', 'strategy', 'km', 'marketplace', 'possess', 'perception', 'artist', 'pumped', 'purposes', 'bent', 'excel', 'dean', 'terribly', 'subsequently', '70s', 'misunderstanding', 'tension', 'pumping', 'wen', 'nation', 'nick', 'rep', 'messing', '##aka', 'unclear', 'nightclub', 'image', 'governor', 'speakers', 'however', '"', 'accounts', 'journey', 'bias', 'position', 'stockholm', 'valerie', 'attempt', 'focusing', 'indicating', 'ru', 'competition', 'very', 'paradox', 'ami', 'expressed', '##hra', '##sten', 'gen', 'forty', 'dumb', '##gur', 'stunning', '##min', '##ise', 'nasty', '##pa', '##hn', '##ai', 'motivated', 'topic', 'rounded', '##tou', 'switched']
    model_folder = f'{get_classification_model_folder()}/model'
    tokenizer = AutoTokenizer.from_pretrained(model_folder, device_map='cuda')
    eval_dataset = ListDataset(torch.load(get_preprocessed_dataset_path('eval'), weights_only=True))
    counts = { candidate: { 0: 0, 1: 0 } for candidate in CANDIDATES }
    for i in tqdm(range(1000)):
        try:
            text = eval_dataset[i]['text']
            label = eval_dataset[i]['label']
            tokens = set(tokenizer.tokenize(text))
            for candidate in CANDIDATES:
                if candidate in tokens:
                    counts[candidate][label] += 1
        except:
            print(f'Error counting index {i}')
    plot_candidate_counts(counts)


def interpret_experiment():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    incriminated_counts = {}
    vindicated_counts = {}
    sum_attributions = {}
    count_attributions = {}
    eval_dataset = ListDataset(torch.load(get_preprocessed_dataset_path('eval'), weights_only=True))
    explainer = get_explainer()
    class_name="LLM-Generated"
    threshold = 0.3
    for i in tqdm(range(1000)):
        try:
            text = eval_dataset[i]['text']
            attributions = explainer(text, class_name=class_name)
            for (token, score) in attributions:
                if score > threshold:
                    if token not in incriminated_counts:
                        incriminated_counts[token] = 0
                    incriminated_counts[token] = incriminated_counts[token] + 1
                elif score < -threshold:
                    if token not in vindicated_counts:
                        vindicated_counts[token] = 0
                    vindicated_counts[token] = vindicated_counts[token] + 1
                if token not in sum_attributions:
                    sum_attributions[token] = 0
                sum_attributions[token] = sum_attributions[token] + score

                if token not in count_attributions:
                    count_attributions[token] = 0
                count_attributions[token] = count_attributions[token] + 1
        except:
            print(f'Error interpreting index {i}')

    average_attributions = { token: sum_attributions[token] / count_attributions[token] for token in sum_attributions.keys() }

    K=50
    top_k_sum_attributions = get_top_k(sum_attributions, K)
    top_k_average_attributions = get_top_k(average_attributions, K)

    print("Top 100 Sum Attributions:")
    print(get_top_k(average_attributions, 100))

    # Unpack the dictionary into two lists
    words = list(top_k_sum_attributions.keys())
    scores = list(top_k_sum_attributions.values())

    # Create the plot for sum attributions
    plt.figure(figsize=(15, 10))
    plt.bar(words, scores, color='skyblue')
    plt.xlabel('Tokens')
    plt.ylabel('Sum Attribution Value')
    plt.title(f'Sum Attribution Value of Top {K} Tokens')
    plt.xticks(rotation=70)  # Rotate the words for better readability
    plt.savefig(f'./interpreter_visualizations/sum_attributions_{class_name.lower()}.png')
    plt.close()

    # Unpack the dictionary into two lists for average attributions
    avg_words = list(top_k_average_attributions.keys())
    avg_scores = list(top_k_average_attributions.values())

    # Create the plot for average attributions
    plt.figure(figsize=(15, 10))
    plt.bar(avg_words, avg_scores, color='lightgreen')
    plt.xlabel('Tokens')
    plt.ylabel('Average Attribution Value')
    plt.title(f'Average Attribution Value of Top {K} Tokens')
    plt.xticks(rotation=70)  # Rotate the words for better readability
    plt.savefig(f'./interpreter_visualizations/average_attributions_{class_name.lower()}.png')
    plt.close()

    # Prepare data for scatter plot
    tokens = list(set(incriminated_counts.keys()).union(set(vindicated_counts.keys())))
    x_values = [incriminated_counts.get(token, 0) for token in tokens]
    y_values = [vindicated_counts.get(token, 0) for token in tokens]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, color='blue')
    plt.xlabel('Incriminated Counts')
    plt.ylabel('Vindicated Counts')
    plt.title('Incriminated vs Vindicated Counts')
    
    # Annotate each point with the token
    for i, token in enumerate(tokens):
        plt.annotate(token, (x_values[i], y_values[i]), fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')

    plt.savefig(f'./interpreter_visualizations/incriminated_vs_vindicated_counts.png')
    plt.close()


def interpret_experiment2():
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    incriminated_counts = {}
    vindicated_counts = {}
    sum_attributions = {}
    count_attributions = {}
    eval_dataset = ListDataset(torch.load(get_preprocessed_dataset_path('eval'), weights_only=True))
    explainer = get_explainer()
    class_name="LLM-Generated"
    threshold = 0.3
    for i in tqdm(range(1000)):
        try:
            text = eval_dataset[i]['text']
            attributions = explainer(text, class_name=class_name)
            for (token, score) in attributions:
                if score > threshold:
                    if token not in incriminated_counts:
                        incriminated_counts[token] = 0
                    incriminated_counts[token] = incriminated_counts[token] + 1
                elif score < -threshold:
                    if token not in vindicated_counts:
                        vindicated_counts[token] = 0
                    vindicated_counts[token] = vindicated_counts[token] + 1
                if token not in sum_attributions:
                    sum_attributions[token] = 0
                sum_attributions[token] = sum_attributions[token] + score

                if token not in count_attributions:
                    count_attributions[token] = 0
                count_attributions[token] = count_attributions[token] + 1
        except:
            print(f'Error interpreting index {i}')

    average_attributions = { token: sum_attributions[token] / count_attributions[token] for token in sum_attributions.keys() }

    K=50
    top_k_sum_attributions = get_top_k(sum_attributions, K)
    top_k_average_attributions = get_top_k(average_attributions, K)

    print("Top 100 Sum Attributions:")
    print(get_top_k(average_attributions, 100))

    # Unpack the dictionary into two lists
    words = list(top_k_sum_attributions.keys())
    scores = list(top_k_sum_attributions.values())

    # Create the plot for sum attributions
    plt.figure(figsize=(15, 10))
    plt.bar(words, scores, color='skyblue')
    plt.xlabel('Tokens')
    plt.ylabel('Sum Attribution Value')
    plt.title(f'Sum Attribution Value of Top {K} Tokens')
    plt.xticks(rotation=70)  # Rotate the words for better readability
    plt.savefig(f'./interpreter_visualizations/sum_attributions_{class_name.lower()}.png')
    plt.close()

    # Unpack the dictionary into two lists for average attributions
    avg_words = list(top_k_average_attributions.keys())
    avg_scores = list(top_k_average_attributions.values())

    # Create the plot for average attributions
    plt.figure(figsize=(15, 10))
    plt.bar(avg_words, avg_scores, color='lightgreen')
    plt.xlabel('Tokens')
    plt.ylabel('Average Attribution Value')
    plt.title(f'Average Attribution Value of Top {K} Tokens')
    plt.xticks(rotation=70)  # Rotate the words for better readability
    plt.savefig(f'./interpreter_visualizations/average_attributions_{class_name.lower()}.png')
    plt.close()

    # Prepare data for scatter plot
    tokens = list(set(incriminated_counts.keys()).union(set(vindicated_counts.keys())))
    x_values = [incriminated_counts.get(token, 0) for token in tokens]
    y_values = [vindicated_counts.get(token, 0) for token in tokens]

    # Create scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, color='blue')
    plt.xlabel('Incriminated Counts')
    plt.ylabel('Vindicated Counts')
    plt.title('Incriminated vs Vindicated Counts')
    
    # Annotate each point with the token
    for i, token in enumerate(tokens):
        plt.annotate(token, (x_values[i], y_values[i]), fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')

    plt.savefig(f'./interpreter_visualizations/incriminated_vs_vindicated_counts.png')
    plt.close()

def get_top_k(values: dict, K: int):
    return dict(sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)[:K])

def plot_score_distributions(values: dict, xlabel: str, ylabel: str, title: str, filename: str):
        # Unpack the dictionary into two lists for average attributions
    x_values = list(values.keys())
    y_balues = list(values.values())

    # Create the plot for average attributions
    plt.figure(figsize=(10, 6))
    plt.bar(x_values, y_balues, color='lightgreen')
    plt.xlabel('Tokens')
    plt.ylabel('Average Attribution Value')
    plt.title('Average Attribution Value of Top 70 Tokens')
    plt.xticks(rotation=45)  # Rotate the words for better readability
    plt.savefig(f'./interpreter_visualizations/{filename}.png')
    plt.close()

def plot_candidate_counts(word_counts):
    # Extract data
    words = list(word_counts.keys())
    positive_counts = [counts[1] for counts in word_counts.values()]
    negative_counts = [counts[0] for counts in word_counts.values()]
    x_positions = np.arange(len(words))  # Positions for tokens on the x-axis
    width = 0.4  # Bar width

    # Create the plot
    plt.figure(figsize=(15, 8))

    # Plot bars
    bars_pos = plt.bar(x_positions, positive_counts, width, color='lightgreen', label='LLM-Generated Count (Label 1)')
    bars_neg = plt.bar(x_positions, [-count for count in negative_counts], width, color='lightcoral', label='Authentic Count (Label 0)')

    # Add lines from bars to words
    for i, bar_neg in enumerate(bars_neg):
        # Line from negative bar to its x-axis label
        plt.plot([x_positions[i], x_positions[i]], [bar_neg.get_height(), -150], color='gray', linestyle='--', linewidth=0.8)

    # Add token names as x-axis labels
    plt.xticks(x_positions, words, rotation=90, fontsize=8)

    # Add labels, title, and legend
    plt.ylabel("Count", fontsize=12)
    plt.xlabel("Tokens", fontsize=12)
    plt.title("Token Counts by Label (LLM-Generated Above, Authentic Below)", fontsize=16)
    plt.axhline(0, color='black', linewidth=0.8)  # Add x-axis line
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'./interpreter_visualizations/candidate_counts.png')
    plt.close()