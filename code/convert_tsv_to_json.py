import csv
import json
from pathlib import Path

# Extract room names from edges.tsv and save as JSON
data_dir = Path('code/data')
edges_path = data_dir / 'edges.tsv'

room_set = set()
with open(edges_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) >= 2:
            room_set.add(row[0].strip().lower())
            room_set.add(row[1].strip().lower())

# Convert to sorted list for deterministic order
room_list = sorted(room_set)

# Save as JSON in the extension folder
extension_dir = Path('deploy_client_extension')
extension_dir.mkdir(exist_ok=True)
rooms_json_path = extension_dir / 'rooms.json'
with open(rooms_json_path, 'w', encoding='utf-8') as f:
    json.dump(room_list, f, indent=2)

print(f"Saved {len(room_list)} room names to {rooms_json_path}")