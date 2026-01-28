import csv
import json

csv_file = 'performance_data.csv'
json_file = 'performance_data.json'
csv_output = 'performance_data_cleaned.csv'
json_output = 'performance_data_cleaned.json'

# Clean CSV
def clean_csv(input_path, output_path):
    with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if all(value.strip() != '' for value in row.values()):
                writer.writerow(row)

# Clean JSON
def clean_json(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    # If the JSON is a list of dicts
    if isinstance(data, list):
        def is_valid(value):
            return value is not None and str(value).strip() != '' and str(value).strip().lower() != 'null'
        cleaned = [row for row in data if all(is_valid(value) for value in row.values())]
    else:
        cleaned = data
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(cleaned, outfile, indent=2)

if __name__ == '__main__':
    clean_csv(csv_file, csv_output)
    clean_json(json_file, json_output)
    print(f'Cleaned CSV written to {csv_output}')
    print(f'Cleaned JSON written to {json_output}')
