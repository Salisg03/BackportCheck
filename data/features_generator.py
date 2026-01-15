import json
import csv
import math
import re
from collections import defaultdict
from datetime import datetime 
import requests 

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

INPUT_FILE =  r"data\raw_data\openstack_all_backport_usage.jsonl" 
OUTPUT_FILE = "openstack_complete.csv"

# REGEX PATTERNS
BUG_PATTERN = re.compile(r"(?:closes-bug|bug|related-bug):\s*#?(\d+)", re.IGNORECASE)
CVE_PATTERN = re.compile(r"(cve-\d{4}-\d+|security|credential|vulnerability)", re.IGNORECASE)
REVERT_PATTERN = re.compile(r"^revert", re.IGNORECASE)
RELATION_PATTERN = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]") 
SUBJECT_TAG_PATTERN = re.compile(r"^\[.*?\]") 

# HISTORY TRACKERS
author_stats = defaultdict(lambda: {'submissions': 0, 'accepted_backports': 0, 'total_churn': 0})
file_stats = defaultdict(lambda: {'touched': 0, 'backported': 0})
project_stats = defaultdict(lambda: {'submissions': 0, 'accepted_backports': 0})

def get_file_ext(filename):
    if filename.startswith("/"): return "none"
    if '.' in filename and not filename.startswith('.'):
        return '.' + filename.split('.')[-1].lower()
    return 'none'

# --- HELPER TO FIND REVISION 1 (NO LEAKAGE) ---
def get_revision_1(change_record):
    """
    Finds the FIRST revision to simulate submission time.
    """
    revisions = change_record.get('revisions', {})
    target_rev = None
    target_rev_id = None
    
    # 1. Try to find explicitly number 1
    for rev_id, rev_data in revisions.items():
        if rev_data.get('_number') == 1:
            target_rev = rev_data
            target_rev_id = rev_id
            break
            
    # 2. Fallback: Take the one with the earliest date or just the first one in list
    if not target_rev and revisions:
        # Just grab the first key available
        target_rev_id = list(revisions.keys())[0]
        target_rev = revisions[target_rev_id]
        
    return target_rev_id, target_rev

# --- LAUNCHPAD API HELPER ---
def get_launchpad_metrics(message):
    match = BUG_PATTERN.search(message)
    if not match: return 0, 0, 0

    bug_id = match.group(1)
    url = f"https://api.launchpad.net/1.0/bugs/{bug_id}"
    try:
        response = requests.get(url, timeout=2)
        if response.status_code == 200:
            data = response.json()
            heat = data.get('heat', 0)
            importance = data.get('importance', 'Undecided')
            severity_map = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1, 'Wishlist': 0, 'Undecided': 0, 'Unknown': 0}
            severity_score = severity_map.get(importance, 0)
            comment_count = data.get('message_count', 0)
            return heat, severity_score, comment_count
    except: pass
    return 0, 0, 0

# AI FEATURES (UPDATED TO USE REV 1) 
def generate_semantic_features(all_data):
    print("Loading AI Model (microsoft/codebert-base)")
    model = SentenceTransformer('bert-base-uncased')
    
    change_ids = []
    messages = []
    
    for change in all_data:
        cid = change.get('change_id')
        
        ### --- MODIFIED: Use Revision 1 ---
        _, rev = get_revision_1(change)
        if not rev: continue
        
        msg = rev.get('commit', {}).get('message', '')
        # Clean message (remove metadata for embedding)
        clean_lines = []
        for line in msg.split('\n'):
            if not re.match(r'^(Change-Id|Signed-off-by|Acked-by|Reviewed-by):', line):
                clean_lines.append(line)
        clean_msg = "\n".join(clean_lines).strip()
        
        change_ids.append(cid)
        messages.append(clean_msg)

    if not messages: return {}, 0

    print(f"Generating Embeddings for {len(messages)} commits...")
    embeddings = model.encode(messages, batch_size=32, show_progress_bar=True)

    n_components = 15 
    print(f"Compressing AI features (768 -> {n_components})...")
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    embedding_map = {}
    for i, cid in enumerate(change_ids):
        embedding_map[cid] = reduced_embeddings[i]
        
    return embedding_map, n_components

# --- TEXT ANALYTICS ---
def count_syllables(word):
    word = word.lower()
    count = 0; vowels = "aeiouy"
    if len(word) == 0: return 0
    if word[0] in vowels: count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels: count += 1
    if word.endswith("e"): count -= 1
    if count == 0: count += 1
    return count

def is_complex_word(word): return count_syllables(word) >= 3

def analyze_text_metrics(message):
    tokens = message.split()
    token_count = len(tokens)
    sentences = [s for s in re.split(r'[.!?]+', message) if s.strip()]
    sentence_count = len(sentences) if len(sentences) > 0 else 1
    avg_sent_len = token_count / sentence_count
    syllable_count = sum(count_syllables(t) for t in tokens)
    complex_word_count = sum(1 for t in tokens if is_complex_word(t))
    
    if token_count > 0:
        flesch = 206.835 - (1.015 * avg_sent_len) - (84.6 * (syllable_count / token_count))
        fog = 0.4 * (avg_sent_len + 100 * (complex_word_count / token_count))
    else: flesch = 0; fog = 0
    return avg_sent_len, flesch, fog

def determine_target(change):
    labels = change.get('labels', {}).get('Backport-Candidate', {})
    final_vote_val = 0
    if 'all' in labels:
        votes = [v for v in labels['all'] if 'date' in v and 'value' in v]
        if votes:
            votes.sort(key=lambda x: x['date'])
            final_vote_val = int(votes[-1].get('value', 0))
    else:
        final_vote_val = int(labels.get('value', 0))
    if change.get('status') == 'MERGED' and final_vote_val > 0: return 1
    return 0

def extract_features():
    print(f"Reading {INPUT_FILE}")
    data = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try: data.append(json.loads(line))
            except: continue

    if not data: return
    print("Sorting data by date...")
    data.sort(key=lambda x: x.get('created', ''))
    
    # 1. GENERATE AI FEATURES (Using Rev 1 now)
    embedding_map, n_ai_features = generate_semantic_features(data)

    print(f"Processing {len(data)} records...")

    headers = [
        'project', 'change_id', 'target', 
        '1_references_bug_tracker', '2_author_success_rate', '3_churn_log_size',
        '5_deletion_ratio', '6_file_count', '8_avg_sentence_length',
        '9_msg_complexity', '10_has_security_impact', '11_nlp_change_type', 
        '12_is_test_change', '13_is_revert', '14_modifies_dependencies',
        '15_author_submission_count', '16_author_trust_score', '17_historical_file_prob',
        '18_is_documentation_only', '19_author_file_confidence', '20_risk_module_coupling', 
        '21_file_extension_entropy', '22_relation_depth', '23_project_acceptance_rate', 
        '24_change_entropy', '25_directory_depth', '26_msg_readability_ease', 
        '27_msg_gunning_fog', '28_has_gerrit_topic', '29_has_subject_tag',
        '31_test_code_ratio', '32_config_change', '33_desc_density', '34_is_weekend',
        '35_bug_heat', '36_bug_severity', '37_bug_comments', '38_is_bot',
        ### --- NEW COLUMN FOR DL ---
        'full_text_dl' 
    ]
    
    for i in range(n_ai_features):
        headers.append(f'40_sem_vec_{i}')

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=headers)
        writer.writeheader()

        for i, change in enumerate(data):
            
            ### --- MODIFIED: Use Revision 1 logic ---
            # We ignore 'current_revision' and calculate explicitly revision 1
            rev_id, revision_info = get_revision_1(change)
            
            if not revision_info: continue
            
            files_data = revision_info.get('files', {}) 
            if not files_data: continue

            commit = revision_info.get('commit', {})
            msg = commit.get('message', '')
            
            # CLEAN MESSAGE FOR DL
            clean_lines = []
            for line in msg.split('\n'):
                if not re.match(r'^(Change-Id|Signed-off-by|Acked-by|Reviewed-by):', line):
                    clean_lines.append(line)
            clean_msg = "\n".join(clean_lines).strip()
            
            subject = change.get('subject', '')
            owner_info = change.get('owner', {})
            author_id = owner_info.get('_account_id', 'unknown')
            author_name = owner_info.get('name', '') 
            project_name = change.get('project', 'unknown')
            change_id = change.get('change_id')
            
            heat, severity, comments = get_launchpad_metrics(msg)
            is_bot = 1 if "OpenStack Proposal Bot" in author_name else 0

            # Date Parsing
            created_str = change.get('created', '')
            is_weekend = 0
            try:
                created_dt = datetime.strptime(created_str.split('.')[0], "%Y-%m-%d %H:%M:%S")
                if created_dt.weekday() >= 5: is_weekend = 1
            except: pass

            # Analyse Fichiers (Using Rev 1 files)
            insertions = 0; deletions = 0
            test_churn = 0
            has_config_file = 0
            file_paths = []
            
            # --- MODIFIED: Generate "Diff Text" for DL ---
            diff_text = "FILES CHANGED:\n"
            
            for fname, fmeta in files_data.items():
                if fname == "/COMMIT_MSG": continue
                churn = 0
                if fmeta:
                    ins = fmeta.get('lines_inserted', 0)
                    rem = fmeta.get('lines_deleted', 0)
                    insertions += ins
                    deletions += rem
                    churn = ins + rem
                    # Add to diff text
                    diff_text += f"{fname} (+{ins}, -{rem})\n"
                
                file_paths.append(fname)
                if "test" in fname.lower() or "zuul.d" in fname: test_churn += churn
                ext = get_file_ext(fname)
                if ext in ['.conf', '.ini', '.yaml', '.json', '.xml']: has_config_file = 1

            total_churn = insertions + deletions
            
            # --- MODIFIED: Combine for DL ---
            full_text_dl = f"{clean_msg}\n[SEP]\n{diff_text}"

            # Feature Calculations
            f31 = test_churn / total_churn if total_churn > 0 else 0
            f32 = has_config_file
            f33 = len(msg) / (total_churn + 1)
            f34 = is_weekend
            f35 = heat
            f36 = severity
            f37 = comments
            f38 = is_bot

            target = determine_target(change)
            avg_sent_len, flesch, gunning_fog = analyze_text_metrics(msg)

            f1 = 1 if BUG_PATTERN.search(msg) else 0
            a_stats = author_stats[author_id]
            f2 = a_stats['accepted_backports'] / a_stats['submissions'] if a_stats['submissions'] > 0 else 0
            f3 = math.log(total_churn + 1)
            f5 = deletions / total_churn if total_churn > 0 else 0
            f6 = len(file_paths)
            f8 = avg_sent_len
            f9 = len(msg)
            f10 = 1 if CVE_PATTERN.search(msg) else 0
            
            msg_lower = msg.lower()
            if "fix" in msg_lower or "bug" in msg_lower: f11 = "Bug Fix"
            elif "doc" in msg_lower: f11 = "Doc"
            elif "test" in msg_lower or "ci" in msg_lower: f11 = "CI"
            else: f11 = "Feature"

            f12 = 1
            for fp in file_paths:
                if not ("test" in fp or "zuul.d" in fp or ".tox" in fp): f12 = 0; break
            
            f13 = 1 if REVERT_PATTERN.match(subject) else 0
            f14 = 0
            for fp in file_paths:
                if "requirements.txt" in fp or "bindep.txt" in fp or "setup.py" in fp: f14 = 1; break

            f15 = a_stats['submissions']
            f16 = a_stats['submissions'] / (a_stats['total_churn'] + 1)
            
            probs = []
            for fp in file_paths:
                fstat = file_stats[fp]
                if fstat['touched'] > 0: probs.append(fstat['backported'] / fstat['touched'])
                else: probs.append(0.0)
            f17 = max(probs) if probs else 0.0

            f18 = 1
            for fp in file_paths:
                if get_file_ext(fp) not in ['.rst', '.md', '.txt']: f18 = 0; break

            f19 = f2 * f17
            depths = [fp.count('/') for fp in file_paths]
            f20 = max(depths) if depths else 0
            exts = [get_file_ext(fp) for fp in file_paths]
            f21 = len(set(exts))
            match = RELATION_PATTERN.search(subject)
            f22 = int(match.group(1)) - 1 if match else (len(commit.get('parents', [])) - 1)
            p_stats = project_stats[project_name]
            f23 = p_stats['accepted_backports'] / p_stats['submissions'] if p_stats['submissions'] > 0 else 0
            
            total_c = 0; file_c = []
            for fmeta in files_data.values():
                if not fmeta: continue
                c = fmeta.get('lines_inserted', 0) + fmeta.get('lines_deleted', 0)
                file_c.append(c); total_c += c
            entropy = 0.0
            if total_c > 0:
                for c in file_c:
                    if c > 0:
                        p = c / total_c
                        entropy -= p * math.log2(p)
            f24 = entropy

            f25 = sum(depths) / len(depths) if depths else 0
            f26 = flesch
            f27 = gunning_fog
            f28 = 1 if change.get('topic') else 0
            f29 = 1 if SUBJECT_TAG_PATTERN.search(subject) else 0

            row = {
                'project': project_name, 'change_id': change_id, 'target': target,
                '1_references_bug_tracker': f1, '2_author_success_rate': round(f2, 4),
                '3_churn_log_size': round(f3, 4), '5_deletion_ratio': round(f5, 4),
                '6_file_count': f6, '8_avg_sentence_length': round(f8, 2),
                '9_msg_complexity': f9, '10_has_security_impact': f10,
                '11_nlp_change_type': f11, '12_is_test_change': f12,
                '13_is_revert': f13, '14_modifies_dependencies': f14,
                '15_author_submission_count': f15, '16_author_trust_score': round(f16, 6),
                '17_historical_file_prob': round(f17, 4), '18_is_documentation_only': f18,
                '19_author_file_confidence': round(f19, 4), '20_risk_module_coupling': int(f20),
                '21_file_extension_entropy': f21, '22_relation_depth': f22,
                '23_project_acceptance_rate': round(f23, 4), '24_change_entropy': round(f24, 4),
                '25_directory_depth': round(f25, 2), '26_msg_readability_ease': round(f26, 2),
                '27_msg_gunning_fog': round(f27, 2), '28_has_gerrit_topic': f28,
                '29_has_subject_tag': f29,
                '31_test_code_ratio': round(f31, 4), '32_config_change': f32,
                '33_desc_density': round(f33, 4), '34_is_weekend': f34,
                '35_bug_heat': int(f35),
                '36_bug_severity': int(f36),
                '37_bug_comments': int(f37),
                '38_is_bot': f38,
                
                ### --- ADDED COLUMN ---
                'full_text_dl': full_text_dl 
            }
            
            # INJECT AI
            vec = embedding_map.get(change_id, [0]*n_ai_features) 
            for dim, val in enumerate(vec):
                row[f'40_sem_vec_{dim}'] = round(val, 6)

            writer.writerow(row)

            # UPDATE HISTORY (Using logic that submission happened)
            author_stats[author_id]['submissions'] += 1
            author_stats[author_id]['total_churn'] += total_churn
            if target == 1: author_stats[author_id]['accepted_backports'] += 1
            for fp in file_paths:
                file_stats[fp]['touched'] += 1
                if target == 1: file_stats[fp]['backported'] += 1
            project_stats[project_name]['submissions'] += 1
            if target == 1: project_stats[project_name]['accepted_backports'] += 1

            if i % 100 == 0: print(f"Processed {i}...", end='\r')

    print(f"\nDone! Features enhanced and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    extract_features()