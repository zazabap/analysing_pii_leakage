import pandas as pd

# Load the subset CSV file
subset_path = "CLERC_subset.csv"
subset_df = pd.read_csv(subset_path)

# Extract the 'query' and 'positive_passages' columns
subset_query_positive = subset_df[['query', 'query_id']]

# Save the extracted data to a new CSV file
output_path = "CLERC_query_positive.csv"
subset_query_positive.to_csv(output_path, index=False, encoding="utf-8")

print(f"Extracted 'query' and 'positive_passages' saved to {output_path}")
