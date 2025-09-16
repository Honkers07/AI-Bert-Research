import json

desc_file = r"C:\Users\wilbe\OneDrive\Desktop\Filtered Patents\descriptions_matched.jsonl" #file link to description
claim_file = r"C:\Users\wilbe\OneDrive\Desktop\Filtered Patents\claims_matched.jsonl" #file link to claim
def load_ids(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return {json.loads(line)["publication_number"] for line in f} #formatting for JSONL file

desc_ids = load_ids(desc_file)
claim_ids = load_ids(claim_file)

print(f"📄 Descriptions: {len(desc_ids):,} unique publication numbers") #confirming that the they are matching
print(f"📄 Claims:       {len(claim_ids):,} unique publication numbers")

# 🔍 Check alignment
only_in_desc = desc_ids - claim_ids
only_in_claim = claim_ids - desc_ids
both = desc_ids & claim_ids

print(f"\n✅ Matched IDs in BOTH: {len(both):,}") #at the end print how many have matched
print(f"⚠️  In descriptions ONLY: {len(only_in_desc):,}")
print(f"⚠️  In claims ONLY:       {len(only_in_claim):,}")
