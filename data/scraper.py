import requests
import json
import time
from collections import Counter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://review.opendev.org/changes/"
OUTPUT_FILE = "openstack_all_backport_usage.jsonl"
BATCH_SIZE = 50 


INCLUDE_ABANDONED = True 


vote_query = (
    "("
    "label:Backport-Candidate=-2 OR "
    "label:Backport-Candidate=-1 OR "
    "label:Backport-Candidate=+1 OR "
    "label:Backport-Candidate=+2"
    ")"
)

if INCLUDE_ABANDONED:
    QUERY = f"{vote_query} (status:merged OR status:abandoned)"
else:
    QUERY = f"{vote_query} status:merged"


OPTIONS = ["SUBMIT_REQUIREMENTS", "DETAILED_LABELS", "CURRENT_REVISION", "CURRENT_COMMIT","CURRENT_FILES"]

def get_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def extract_all_usage():
    print(f"STARTING ALL USAGE EXTRACTION ")
    print(f"Logic:   Save ALL changes with Backport-Candidate votes (No strict filtering)")
    print(f"Query:   {QUERY}")
    print(f"Output:  {OUTPUT_FILE}")


    session = get_session()
    start = 0
    total_saved = 0
    
    # Track which projects these come from
    project_counter = Counter()

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        while True:
            params = {"q": QUERY, "o": OPTIONS, "n": BATCH_SIZE, "S": start}

            try:
                response = session.get(BASE_URL, params=params, timeout=60)
                response.raise_for_status()
            except Exception as e:
                print(f"\n[!] Connection error: {e}. Retrying in 10s...")
                time.sleep(10)
                continue

            content = response.text
            if content.startswith(")]}'"): content = content[4:]

            try:
                changes = json.loads(content)
            except:
                print(f"\n[!] JSON Error. Skipping batch.")
                start += BATCH_SIZE
                continue

            if not changes:
                print("\nReached end of results.")
                break

            batch_saved = 0
            
            for change in changes:
                # 1. Track Project Stats
                proj = change.get('project', 'unknown')
                project_counter[proj] += 1
                f.write(json.dumps(change) + "\n")
                
                batch_saved += 1
                total_saved += 1

            # Progress Bar
            print(f"Saved: {total_saved} records (Last batch: {batch_saved})", end='\r')

            # Pagination
            if '_more_changes' in changes[-1]:
                start += BATCH_SIZE
                time.sleep(0.5)
            else:
                if len(changes) < BATCH_SIZE: break
                start += BATCH_SIZE
                time.sleep(0.5)

    print(f"EXTRACTION COMPLETE.")
    print(f"Total Changes Saved: {total_saved}")
    print("TOP 20 PROJECTS IN DATASET:")
    for proj, count in project_counter.most_common(20):
        print(f"{count:4d} | {proj}")

    # Save stats to file
    with open("project_usage_stats.txt", "w") as pf:
        pf.write("Count | Project Name\n")
        pf.write("--------------------\n")
        for proj, count in project_counter.most_common():
            pf.write(f"{count:5d} | {proj}\n")

if __name__ == "__main__":
    extract_all_usage()