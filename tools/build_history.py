import json
from collections import defaultdict

INPUT_FILE = r"data\raw_data\openstack_all_backport_usage.jsonl" 
OUTPUT_FILE = "stats_complete.json"

stats = {
    "authors": defaultdict(lambda: {'submissions': 0, 'accepted_backports': 0, 'total_churn': 0}),
    "files": defaultdict(lambda: {'touched': 0, 'backported': 0}),
    "projects": defaultdict(lambda: {'submissions': 0, 'accepted_backports': 0}),
    "last_updated": "2023-01-01 00:00:00"
}

print("construction de l'historique...")
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            change = json.loads(line)
            
            # Date
            created = change.get('created', '')
            if created > stats['last_updated']: stats['last_updated'] = created

            # Target (Logique stricte training)
            labels = change.get('labels', {}).get('Backport-Candidate', {})
            success = 0
            if 'all' in labels:
                votes = [v for v in labels['all'] if 'value' in v]
                if votes:
                    votes.sort(key=lambda x: x.get('date', ''))
                    if int(votes[-1].get('value', 0)) > 0: success = 1
            elif int(labels.get('value', 0)) > 0: success = 1
            
            owner = str(change.get('owner', {}).get('_account_id', 'unknown'))
            project = change.get('project', 'unknown')
            
            # Files
            rev = list(change.get('revisions', {}).values())[0]
            files = rev.get('files', {})
            churn = sum(m.get('lines_inserted',0)+m.get('lines_deleted',0) for f,m in files.items() if f != "/COMMIT_MSG")

            # Update
            stats["authors"][owner]['submissions'] += 1
            stats["authors"][owner]['total_churn'] += churn
            if success: stats["authors"][owner]['accepted_backports'] += 1
            
            stats["projects"][project]['submissions'] += 1
            if success: stats["projects"][project]['accepted_backports'] += 1
            
            for fp in files:
                if fp == "/COMMIT_MSG": continue
                stats["files"][fp]['touched'] += 1
                if success: stats["files"][fp]['backported'] += 1

    # Conversion defaultdict -> dict pour JSON
    final_stats = {
        "meta_last_updated": stats['last_updated'],
        "authors": dict(stats['authors']),
        "files": dict(stats['files']),
        "projects": dict(stats['projects'])
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_stats, f)
    print("stats_complete.json régénéré.")

except FileNotFoundError:
    print(f"Fichier {INPUT_FILE} introuvable.")