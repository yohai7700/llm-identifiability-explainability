import csv
from torch.utils.data import Dataset

# Save dataset to CSV
def save_dataset_to_csv(dataset, file_path):
    # Get keys from the first item to define the CSV columns
    keys = list(dataset[0].keys())
    keys.append('index')
    
    # Open CSV file for writing
    with open(file_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        
        # Write the header (column names)
        writer.writeheader()
        
        success_count = 0
        # Write each row (dictionary)
        for i in range(len(dataset)):
            try:
                item = dataset[i]
                item.update({ 'index': i })
                writer.writerow(item)

                success_count += 1
            except Exception as e:
                print(f"Error writing item {i}: {e}")

        print(f"Successfully wrote {success_count} items to {file_path} out of {len(dataset)} items.")