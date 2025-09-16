file_path = r"C:\Users\wilbe\OneDrive\Desktop\g02b_matched_pub_ids_clean.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        print(f"Line {i}: {line.strip()}")
        if i >= 5:
            break