import os
import json
from tqdm import tqdm

# === CONFIG ===
PATENT_DIR = r"C:\Users\wilbe\OneDrive\Desktop\Patent Description"  # Change if needed
OUTPUT_FILE = os.path.join(PATENT_DIR, "g02b_description_filtered_slim.jsonl")
MAX_RESULTS = None  # Optional limit (e.g., 10000)

G02B_KEYWORDS = [
    "optical", "lens", "mirror", "fiber optic", "waveguide",
    "hologram", "prism", "projection", "refraction", "light beam",
    "optoelectronic", "diffraction", "image processing", "optical signal",
    "telecentric", "laser optics", "collimator", "birefringence"
]

# === Extract English long descriptions ===
def extract_descriptions(patent):
    if "description_localized" in patent:
        return [
            d.get("text", "") for d in patent["description_localized"]
            if d.get("language") == "en" and not d.get("truncated", False)
        ]
    elif "description" in patent and isinstance(patent["description"], str):
        return [patent["description"]]
    elif "full_text" in patent and isinstance(patent["full_text"], dict):
        desc = patent["full_text"].get("description")
        if isinstance(desc, str):
            return [desc]
        elif isinstance(desc, list):
            return [str(d) for d in desc]
    return []

# === Extract English title (best-effort) ===
def extract_title(patent):
    if "title_localized" in patent:
        for title_entry in patent["title_localized"]:
            if title_entry.get("language") == "en":
                return title_entry.get("text", "")
        return patent["title_localized"][0].get("text", "")
    elif "title" in patent:
        return str(patent["title"])
    return ""

# === Return True if any G02B keyword appears in the description ===
def filter_description_keywords(patent):
    descriptions = extract_descriptions(patent)
    for desc in descriptions:
        text = desc.lower()
        if any(keyword in text for keyword in G02B_KEYWORDS):
            return desc  # Return the matched description
    return None

# === MAIN ===
def filter_by_description():
    count = 0
    seen_publications = set()
    files = [f for f in os.listdir(PATENT_DIR) if f.endswith(".json")]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_file:
        for fname in tqdm(files, desc="Scanning patents"):
            fpath = os.path.join(PATENT_DIR, fname)

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            patent = json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"❌ JSON decode error in {fname} (line {line_num}): {e}")
                            continue

                        pub_id = patent.get("publication_number")
                        if not pub_id or pub_id in seen_publications:
                            continue

                        if MAX_RESULTS and count >= MAX_RESULTS:
                            break

                        matched_description = filter_description_keywords(patent)
                        if matched_description:
                            output_record = {
                                "publication_number": pub_id,
                                "title": extract_title(patent),
                                "description": matched_description
                            }
                            out_file.write(json.dumps(output_record) + "\n")
                            seen_publications.add(pub_id)
                            count += 1

            except Exception as e:
                print(f"❌ Error reading {fname}: {e}")
                continue

    print(f"\n✅ Done. Saved {count} unique G02B-related patents with description to: {OUTPUT_FILE}")

# === RUN ===
if __name__ == "__main__":
    filter_by_description()