import json
from collections import defaultdict

INPUT_FILE = r"data\raw_data\openstack_all_backport_usage.jsonl"
OUTPUT_FILE = "stats_complete.json"

stats = {
    "authors": defaultdict(lambda: {'total': 0, 'accepted': 0, 'cumulative_churn': 0}),
    "files": defaultdict(lambda: {'total': 0, 'accepted': 0}),
    "projects": defaultdict(lambda: {'total': 0, 'accepted': 0}),
    "last_updated": "2020-01-01 00:00:00"
}

print("1. Building History (Strict Last Vote Logic)...")
count = 0

try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                change = json.loads(line)
                
                # 1. Update Timestamp
                created = change.get('created', '')
                if created > stats['last_updated']: stats['last_updated'] = created

                # 2. Determine Success (Strict Last Vote)
                labels = change.get('labels', {}).get('Backport-Candidate', {})
                votes = labels.get('all', [])
                valid_votes = [v for v in votes if 'value' in v]
                
                if not valid_votes: continue # Skip if no votes
                
                # Sort by date
                valid_votes.sort(key=lambda x: x.get('date', ''))
                final_val = int(valid_votes[-1].get('value', 0))
                
                success = 1 if final_val >= 1 else 0
                
                # 3. Extract Metadata
                owner = str(change.get('owner', {}).get('_account_id', 'unknown')) # Use ID for history script if consistent, or name
                # Ideally, match app.py which uses Name. Let's switch to Name to match app.py
                owner_name = change.get('owner', {}).get('name', 'Unknown')
                
                project = change.get('project', 'unknown')
                
                # 4. Process Files
                rev_data = change.get('revisions', {})
                if not rev_data: continue
                rev = list(rev_data.values())[0]
                files = rev.get('files', {})
                
                churn = 0
                file_list = []
                for f_path, m in files.items():
                    if f_path == "/COMMIT_MSG": continue
                    churn += m.get('lines_inserted',0) + m.get('lines_deleted',0)
                    file_list.append(f_path)

                count += 1

                # 5. Update Stats
                # Author
                stats["authors"][owner_name]['total'] += 1
                stats["authors"][owner_name]['cumulative_churn'] += churn
                if success: stats["authors"][owner_name]['accepted'] += 1
                
                # Project
                stats["projects"][project]['total'] += 1
                if success: stats["projects"][project]['accepted'] += 1
                
                # Files
                for fp in file_list:
                    stats["files"][fp]['total'] += 1
                    if success: stats["files"][fp]['accepted'] += 1
            except Exception as e:
                continue

    final_stats = {
        "meta_last_updated": stats['last_updated'],
        "authors": dict(stats['authors']),
        "files": dict(stats['files']),
        "projects": dict(stats['projects'])
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_stats, f)
        
    print(f"Done. Processed {count} changes. Saved to {OUTPUT_FILE}.")

except FileNotFoundError:
    print(f"Error: Could not find {INPUT_FILE}")