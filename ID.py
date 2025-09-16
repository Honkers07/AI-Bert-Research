import json

input_path = r"C:\Users\wilbe\OneDrive\Desktop\g02b_matched_pub_ids.txt"
output_path = r"C:\Users\wilbe\OneDrive\Desktop\g02b_matched_pub_ids_clean.jsonl"

with open(input_path, "r", encoding="utf-8") as infile, \
     open(output_path, "w", encoding="utf-8") as outfile:
    
    cleaned = 0
    skipped = 0

    for i, line in enumerate(infile, 1):
        line = line.strip()
        if not line:
            skipped += 1
            continue

        try:
            # Try parsing if it's already JSON
            record = json.loads(line)
            pub_id = record.get("publication_number", "").strip()
            if not pub_id:
                skipped += 1
                continue
        except json.JSONDecodeError:
            # Assume it's just a plain ID
            pub_id = line

        if pub_id:
            json.dump({ "publication_number": pub_id }, outfile)
            outfile.write("\n")
            cleaned += 1
        else:
            skipped += 1

print(f"✅ Cleaned {cleaned:,} lines.")
print(f"⚠️ Skipped {skipped:,} invalid/empty lines.")
print(f"📄 Output written to: {output_path}")
