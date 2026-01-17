import json
import re
import math
import pandas as pd
import requests
from collections import defaultdict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import textstat

class FeatureEngineer:
    def __init__(self):
        # --- REGEX PATTERNS ---
        self.bug_id_pattern = re.compile(r"(?:Closes-Bug|Related-Bug|Bug):\s*#?(\d+)", re.IGNORECASE)
        self.security_pattern = re.compile(r"(CVE-\d+|Security|Vulnerability|Credential)", re.IGNORECASE)
        self.revert_pattern = re.compile(r"^Revert\s+\"", re.IGNORECASE)
        self.tag_pattern = re.compile(r"\[.*?\]") 
        self.bot_pattern = re.compile(r"\b(bot|zuul|jenkins|proposal)\b", re.IGNORECASE)
        
        # File Categories
        self.test_paths = ['test', 'tests', 'testing', 'zuul.d', '.zuul.yaml']
        self.dep_files = ['requirements.txt', 'test-requirements.txt', 'bindep.txt', 'setup.py']
        
        # Project Categories
        self.deploy_projects = {
            'openstack/kolla', 'openstack/kolla-ansible', 'openstack/kayobe', 
            'openstack/tripleo-heat-templates', 'openstack/puppet-openstack-integration',
            'openstack/openstack-ansible', 'openstack/bifrost'
        }
        
        # --- ROBUST API SESSION (NEW) ---
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        
        self.bug_cache = {}

    def calculate_entropy(self, files_dict):
        file_churns = []
        for f_path, stats in files_dict.items():
            if f_path == "/COMMIT_MSG": continue
            churn = stats.get('lines_inserted', 0) + stats.get('lines_deleted', 0)
            if churn > 0: file_churns.append(churn)
        
        total_churn = sum(file_churns)
        if total_churn == 0: return 0.0
            
        entropy = 0.0
        for churn in file_churns:
            p = churn / total_churn
            entropy -= p * math.log2(p)
        return entropy

    def get_bug_metadata(self, bug_id):
        if bug_id in self.bug_cache: return self.bug_cache[bug_id]
        url = f"https://api.launchpad.net/1.0/bugs/{bug_id}"
        
        try:
            # Increased timeout and using the robust session
            resp = self.session.get(url, timeout=5) 
            if resp.status_code == 200:
                data = resp.json()
                
                importance = data.get('importance', 'Undecided')
                severity_map = {'Critical': 3, 'High': 2, 'Medium': 1, 'Low': 0, 'Undecided': 0, 'Wishlist': 0}
                severity = severity_map.get(importance, 0)
                
                heat = data.get('heat', 0)
                comments = data.get('message_count', 0)
                tags = data.get('tags', [])
                is_regression = 1 if 'regression' in tags else 0
                
                result = {'severity': severity, 'heat': heat, 'comments': comments, 'is_regression': is_regression}
                self.bug_cache[bug_id] = result
                return result
        except Exception: pass
        
        return {'severity': 0, 'heat': 0, 'comments': 0, 'is_regression': 0}

    def extract_basic_features(self, change_data):
        revision_info = change_data.get('revisions', {})
        latest_commit_hash = change_data.get('current_revision')
        
        if not revision_info or not latest_commit_hash: return None
            
        rev_data = revision_info[latest_commit_hash]
        commit = rev_data.get('commit', {})
        files = rev_data.get('files', {})
        msg = commit.get('message', "")
        subject = change_data.get('subject', "").lower()
        project_name = change_data.get('project', '')
        author_name = commit.get('author', {}).get('name', "Unknown")
        
        # --- 1. MESSAGE METRICS ---
        feat_word_count = textstat.lexicon_count(msg, removepunct=True)
        feat_flesch_ease = textstat.flesch_reading_ease(msg)
        feat_gunning_fog = textstat.gunning_fog(msg)

        # --- 2. BUG METRICS (SAFE VERSION - NO API CALLS) ---
        # Only check if a bug ID is mentioned in the text (Static Analysis)
        match = self.bug_id_pattern.search(msg)
        feat_references_bug_tracker = 1 if match else 0
        
        # REMOVED LEAKAGE FEATURES: severity, heat, comments, regression status
        # (These require querying the live API which leaks future data)

        # --- 3. INTENT CLASSIFICATION ---
        ci_keywords = ['ci', 'gate', 'pipeline', 'job', 'workflow', 'tox', 'lint', 'zuul', 'playbook']
        feat_is_ci_intent = 1 if any(k in subject for k in ci_keywords) else 0
        
        feat_is_feature = 1 if re.search(r'\b(add|implement|support|introduce|new|feat|enable|allow|provide)\b', subject) else 0
        feat_is_refactor = 1 if re.search(r'\b(refactor|clean|remove|move|rename|delete|drop)\b', subject) else 0
        feat_is_fix = 1 if re.search(r'\b(fix|resolve|repair|patch|correct|handle|mitigate|prevent)\b', subject) else 0
        feat_is_maintenance = 1 if re.search(r'\b(update|bump|upgrade|downgrade|pin|unpin|sync)\b', subject) else 0
        feat_is_deploy = 1 if re.search(r'\b(config|conf|deploy|install|set|use|default|variable|param|role)\b', subject) else 0

        # --- 4. FILE ANALYSIS ---
        file_list = [f for f in files.keys() if f != "/COMMIT_MSG"]
        feat_file_count = len(file_list)
        
        config_lines = 0; code_lines = 0; total_lines = 0
        feat_modifies_config = 0; feat_modifies_migration = 0
        feat_modifies_api = 0; feat_modifies_deps = 0; feat_is_ci_file_change = 0
        
        config_exts = {'.yaml', '.yml', '.json', '.ini', '.conf', '.toml', '.xml', '.j2', '.rst', '.md', '.erb'}
        code_exts = {'.py', '.c', '.h', '.cpp', '.java', '.go', '.sh', '.js', '.ts', '.pp'} 
        ci_files_list = ['.zuul.yaml', 'zuul.d', '.gitlab-ci.yml', '.travis.yml', 'tox.ini', 'bindep.txt']

        for f_path in file_list:
            f_lower = f_path.lower()
            stats = files[f_path]
            lines_changed = stats.get('lines_inserted', 0) + stats.get('lines_deleted', 0)
            total_lines += lines_changed
            
            if 'conf' in f_lower or '.ini' in f_lower or '.yaml' in f_lower: feat_modifies_config = 1
            if 'alembic' in f_lower or 'migration' in f_lower or 'upgrade' in f_lower: feat_modifies_migration = 1
            if 'api/' in f_lower or 'v1/' in f_lower or 'v2/' in f_lower: feat_modifies_api = 1
            if any(df in f_lower for df in self.dep_files): feat_modifies_deps = 1
            if any(cif in f_lower for cif in ci_files_list): feat_is_ci_file_change = 1
            
            ext = "." + f_path.split('.')[-1].lower() if '.' in f_path else ""
            if ext in config_exts: config_lines += lines_changed
            elif ext in code_exts: code_lines += lines_changed

        if total_lines > 0:
            feat_config_ratio = config_lines / total_lines
            feat_code_ratio = code_lines / total_lines
        else:
            feat_config_ratio = 0.0; feat_code_ratio = 0.0
            
        feat_is_ci_change = 1 if (feat_is_ci_file_change == 1 or feat_is_ci_intent == 1) else 0
        feat_is_pure_config = 1 if (feat_config_ratio > 0.99) else 0

        # --- 5. ENTROPY & DENSITY ---
        feat_entropy = self.calculate_entropy(files)
        churn_raw = total_lines
        feat_churn_density = churn_raw / feat_file_count if feat_file_count > 0 else 0.0
        feat_churn_log = math.log(churn_raw + 1)
        total_deletions = sum(f.get('lines_deleted', 0) for f in files.values())
        feat_deletion_ratio = total_deletions / churn_raw if churn_raw > 0 else 0.0

        if file_list:
            depths = [f.count('/') + 1 for f in file_list]
            feat_dir_depth = sum(depths) / len(depths)
        else:
            feat_dir_depth = 0
            
        feat_ext_entropy = len(set([f.split('.')[-1] for f in file_list if '.' in f]))
        feat_is_deploy_project = 1 if any(dp in project_name for dp in self.deploy_projects) else 0
        feat_is_bot = 1 if self.bot_pattern.search(author_name) else 0

        # --- 6. INTERACTIONS (SAFE ONLY) ---
        # "Safe Scatter": High entropy is fine if it's pure config
        feat_safe_entropy = feat_entropy * feat_is_pure_config
        
        # REMOVED LEAKAGE INTERACTIONS: critical_bug, regression_fix

        return {
            "change_id": change_data.get('id'),
            "project": change_data.get('project'),
            "author_name": author_name,
            "created_date": change_data.get('created'),
            "target": self.get_target_label(change_data),
            "file_list": file_list, 
            
            # --- FEATURE INTERACTIONS ---
            "safe_entropy_interaction": feat_safe_entropy,
            
            # --- CONTEXT ---
            "is_bot": feat_is_bot,
            "is_deployment_project": feat_is_deploy_project,
            "is_pure_config": feat_is_pure_config,

            # --- INTENT ---
            "is_fix": feat_is_fix,
            "is_feature": feat_is_feature,
            "is_maintenance": feat_is_maintenance,
            "is_deployment": feat_is_deploy,
            "is_ci_change": feat_is_ci_change,
            "is_refactor": feat_is_refactor,
            
            # --- CONTENT ---
            "config_line_ratio": feat_config_ratio,
            "code_line_ratio": feat_code_ratio,
            "churn_density": feat_churn_density,
            "change_entropy": feat_entropy,
            "file_count": feat_file_count,
            "churn_log_size": feat_churn_log,
            "deletion_ratio": feat_deletion_ratio,
            
            # --- TEXT ---
            "msg_readability_ease": feat_flesch_ease,
            "msg_gunning_fog": feat_gunning_fog,
            "references_bug_tracker": feat_references_bug_tracker,
            "has_subject_tag": 1 if self.tag_pattern.search(subject) else 0,
            
            # --- RISKS ---
            "modifies_db_migration": feat_modifies_migration,
            "modifies_dependencies": feat_modifies_deps,
            "modifies_config": feat_modifies_config,
            "modifies_public_api": feat_modifies_api,
            "has_security_impact": 1 if self.security_pattern.search(msg) else 0,
            "is_test_change": 1 if all(tp in f for f in file_list for tp in self.test_paths) else 0,
            "is_revert": 1 if self.revert_pattern.match(change_data.get('subject', '')) else 0,
            "is_documentation_only": 1 if all((f.endswith('.rst') or f.endswith('.md') or 'doc/' in f) for f in file_list) else 0,
            "directory_depth": feat_dir_depth,
            "file_extension_entropy": feat_ext_entropy,
            "has_gerrit_topic": 1 if change_data.get('topic') else 0
        }

    def get_target_label(self, change_data):
        """
        Determines if the change was accepted (1) or rejected (0).
        Logic: Takes the LATEST vote based on time, not the maximum.
        """
        labels = change_data.get('labels', {})
        bc_label = labels.get('Backport-Candidate', {})
        
        
        
        votes = bc_label.get('all', [])
        if not votes:
            return 0 # No votes = No backport
            
        # 2. Filter votes that have a date (should be all of them with DETAILED_LABELS)
        valid_votes = [v for v in votes if 'date' in v]
        
        if not valid_votes:
            # Fallback: If no dates are found, we can't determine the last one.
            # We treat ambiguous cases as Rejected (0) to be safe/strict.
            return 0
            
        # 3. Find the vote with the MAXIMUM (Latest) date string
        # Gerrit dates are ISO formatted (YYYY-MM-DD...), so string sort works perfectly.
        last_vote = max(valid_votes, key=lambda x: x['date'])
        
        # 4. Check the value of the last vote
        # +1 or +2 means Accepted. 0, -1, -2 means Rejected.
        return 1 if last_vote.get('value', 0) >= 1 else 0

def process_history_and_save():
    engineer = FeatureEngineer()
    raw_data = []
    
    input_file = r"data\raw_data\openstack_all_backport_usage.jsonl"
    print("1. Extracting Features (Includes API Retries & Interactions)...")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    change = json.loads(line)
                    feats = engineer.extract_basic_features(change)
                    if feats: raw_data.append(feats)
                except: continue
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    df = pd.DataFrame(raw_data)
    df['created_date'] = pd.to_datetime(df['created_date'])
    df = df.sort_values('created_date').reset_index(drop=True)

    print("2. Calculating Historical Context (Time Travel)...")
    
    author_stats = defaultdict(lambda: {'total': 0, 'accepted': 0, 'cumulative_churn': 0})
    project_stats = defaultdict(lambda: {'total': 0, 'accepted': 0})
    file_stats = defaultdict(lambda: {'total': 0, 'accepted': 0})
    
    success_rates = []; sub_counts = []; trust_scores = []
    proj_accept_rates = []; hist_file_probs = []

    for idx, row in df.iterrows():
        auth = row['author_name']
        proj = row['project']
        target = row['target']
        churn = math.exp(row['churn_log_size']) - 1 
        files = row['file_list']
        
        s = author_stats[auth]
        success_rates.append(s['accepted'] / s['total'] if s['total'] > 0 else 0.0)
        sub_counts.append(s['total'])
        trust_scores.append(s['total'] / (s['cumulative_churn'] + 1))
        
        p = project_stats[proj]
        proj_accept_rates.append(p['accepted'] / p['total'] if p['total'] > 0 else 0.0)
        
        probs = []
        for f in files:
            fs = file_stats[f]
            probs.append(fs['accepted'] / fs['total'] if fs['total'] > 0 else 0.0)
        hist_file_probs.append(max(probs) if probs else 0.0)
        
        s['total'] += 1; s['cumulative_churn'] += churn
        p['total'] += 1
        if target == 1: s['accepted'] += 1; p['accepted'] += 1
            
        for f in files:
            file_stats[f]['total'] += 1
            if target == 1: file_stats[f]['accepted'] += 1

    df['author_success_rate'] = success_rates
    df['author_submission_count'] = sub_counts
    df['author_trust_score'] = trust_scores
    df['project_acceptance_rate'] = proj_accept_rates
    df['historical_file_prob'] = hist_file_probs
    
    df = df.drop(columns=['file_list', 'created_date'])
    
    output_csv = r"data\processed_data\openstack_complete.csv"
    df.to_csv(output_csv, index=False)
    print(f"Done. Saved dataset with {df.shape[1]} columns to {output_csv}")

if __name__ == "__main__":
    process_history_and_save()