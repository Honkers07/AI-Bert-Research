import json

# === File paths ===
CLAIMS_FILE = r"C:\Users\wilbe\OneDrive\Desktop\Filtered Patents\FILTERED_CLAIMS.json"
DESCRIPTIONS_FILE = r"C:\Users\wilbe\OneDrive\Desktop\Filtered Patents\g02b_description_filtered_Description.jsonl"

OUTPUT_CLAIMS = r"C:\Users\wilbe\OneDrive\Desktop\Patents\claims_matched.jsonl"
OUTPUT_DESCRIPTIONS = r"C:\Users\wilbe\OneDrive\Desktop\Patents\descriptions_matched.jsonl"

# === Step 1: Load all claim IDs into a set ===
print("🔍 Loading claim IDs...")
claim_ids = set()
with open(CLAIMS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            obj = json.loads(line)
            pid = obj.get("publication_number")
            if pid:
                claim_ids.add(pid)
        except json.JSONDecodeError:
            continue

print(f"✅ Total claims loaded: {len(claim_ids):,}")

# === Step 2: Filter descriptions to only those with a matching claim ===
print("🔄 Filtering matching descriptions...")
matched_ids = set()
with open(OUTPUT_DESCRIPTIONS, "w", encoding="utf-8") as out_desc, \
     open(DESCRIPTIONS_FILE, "r", encoding="utf-8") as in_desc:

    for line in in_desc:
        try:
            obj = json.loads(line)
            pid = obj.get("publication_number")
            if pid and pid in claim_ids:
                out_desc.write(json.dumps(obj) + "\n")
                matched_ids.add(pid)
        except json.JSONDecodeError:
            continue

print(f"✅ Descriptions matched: {len(matched_ids):,}")

# === Step 3: Write matching claims for those same IDs ===
print("📝 Writing matching claims...")
with open(OUTPUT_CLAIMS, "w", encoding="utf-8") as out_claims, \
     open(CLAIMS_FILE, "r", encoding="utf-8") as in_claims:

    for line in in_claims:
        try:
            obj = json.loads(line)
            pid = obj.get("publication_number")
            if pid in matched_ids:
                out_claims.write(json.dumps(obj) + "\n")
        except json.JSONDecodeError:
            continue

print("\n✅ DONE.")
print(f"📄 Saved {len(matched_ids):,} matched patents to:")
print(f"   • {OUTPUT_DESCRIPTIONS}")
print(f"   • {OUTPUT_CLAIMS}")