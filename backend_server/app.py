import os
import json
import time
import threading
import re
import math
import requests
import textstat
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict
from groq import Groq
from dotenv import load_dotenv

# Load environment variables (API Keys)
load_dotenv()

# --- CONFIGURATION ---
class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Files
    MODEL_PATH = os.path.join(BASE_DIR, "Xgboost_optimized.json") 
    STATS_PATH = os.path.join(BASE_DIR, "stats_complete.json")
    THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.txt")
    
    # Settings
    GERRIT_API = "https://review.opendev.org"
    UPDATE_INTERVAL = 3600  # Update history every hour
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- 1. HISTORY MANAGER (The "Memory" of the system) ---
class HistoryManager:
    def __init__(self):
        # Structure matches the training script logic
        self.stats = {
            "authors": defaultdict(lambda: {'total': 0, 'accepted': 0, 'cumulative_churn': 0}),
            "files": defaultdict(lambda: {'total': 0, 'accepted': 0}),
            "projects": defaultdict(lambda: {'total': 0, 'accepted': 0}),
            "last_updated": "2020-01-01 00:00:00"
        }
        self.load_from_disk()
    
    def load_from_disk(self):
        """Loads historical stats from JSON to initialize the system."""
        if os.path.exists(Config.STATS_PATH):
            try:
                with open(Config.STATS_PATH, "r") as f:
                    data = json.load(f)
                    # Merge loaded data into defaultdicts to prevent KeyErrors
                    for k, v in data.get("authors", {}).items(): self.stats["authors"][k] = v
                    for k, v in data.get("files", {}).items(): self.stats["files"][k] = v
                    for k, v in data.get("projects", {}).items(): self.stats["projects"][k] = v
                    self.stats["last_updated"] = data.get("meta_last_updated", self.stats["last_updated"])
                print(f"[History] Stats loaded. Last update: {self.stats['last_updated']}")
            except Exception as e: 
                print(f"⚠ [History] Load Error: {e}")
                # If error, we start with empty stats (safe fallback)

    def save_to_disk(self):
        """Saves current memory to disk so we don't lose progress on restart."""
        snapshot = {
            "meta_last_updated": self.stats["last_updated"],
            "authors": dict(self.stats["authors"]),
            "files": dict(self.stats["files"]),
            "projects": dict(self.stats["projects"])
        }
        try:
            with open(Config.STATS_PATH, "w") as f: json.dump(snapshot, f)
        except Exception as e: print(f"⚠ [History] Save Error: {e}")

    def fetch_and_update(self):
        """Background task: Periodically fetches new changes from Gerrit to update trust scores."""
        while True:
            try:
                raw_date = self.stats["last_updated"]
                clean_date = raw_date.split('.')[0] # Remove nanoseconds for API
                
                print(f"[Updater] Scanning for closed changes since {clean_date}...")
                
                # Fetch only CLOSED changes that have Backport-Candidate votes
                vote_filter = "(label:Backport-Candidate=-2 OR label:Backport-Candidate=-1 OR label:Backport-Candidate=+1 OR label:Backport-Candidate=+2)"
                query = f'status:closed after:"{clean_date}" AND {vote_filter}'
                
                resp = requests.get(
                    f"{Config.GERRIT_API}/changes/",
                    params={'q': query, 'o': ['CURRENT_REVISION', 'CURRENT_FILES', 'DETAILED_LABELS', 'DETAILED_ACCOUNTS']},
                    timeout=30
                )
                
                if resp.status_code == 200:
                    # Remove magic prefix usually found in Gerrit responses
                    text = resp.text[4:] if resp.text.startswith(")]}'") else resp.text
                    changes = json.loads(text)
                    if changes:
                        self._process_changes(changes)
                    else:
                        print("[Updater] No new relevant changes found.")
                else:
                    print(f"[Updater] Gerrit API Error: {resp.status_code}")
            except Exception as e:
                print(f"[Updater] Exception: {e}")
            
            time.sleep(Config.UPDATE_INTERVAL)

    def _process_changes(self, changes):
        """
        Updates internal stats based on new data.
        CRITICAL: Uses STRICT LAST VOTE logic (ignoring 'approved' tag).
        """
        count = 0
        new_last_date = self.stats["last_updated"]
        
        for change in changes:
            # 1. Update Timestamp
            updated = change.get('updated', change.get('created'))
            if updated > new_last_date: new_last_date = updated
            
            # 2. Determine Success (STRICT LOGIC)
            labels = change.get('labels', {})
            bc_label = labels.get('Backport-Candidate', {})
            
            votes = bc_label.get('all', [])
            # Filter for votes with dates
            valid_votes = [v for v in votes if 'date' in v and 'value' in v]
            
            if not valid_votes: continue # Skip if no clear vote history
            
            # Sort by date to find the final human decision
            valid_votes.sort(key=lambda x: x.get('date', ''))
            final_val = int(valid_votes[-1].get('value', 0))
            
            is_accepted = 1 if final_val >= 1 else 0

            # 3. Extract Metadata
            owner_name = change.get('owner', {}).get('name', 'Unknown')
            project = change.get('project', 'unknown')
            rev_id = change.get('current_revision')
            if not rev_id: continue
            
            files_dict = change['revisions'][rev_id].get('files', {})
            
            # 4. Calculate Churn
            churn = 0
            file_list = []
            for f_path, meta in files_dict.items():
                if f_path == "/COMMIT_MSG": continue
                churn += meta.get('lines_inserted', 0) + meta.get('lines_deleted', 0)
                file_list.append(f_path)
            
            # 5. Update Stats in Memory
            # Author
            self.stats["authors"][owner_name]['total'] += 1
            self.stats["authors"][owner_name]['cumulative_churn'] += churn
            if is_accepted: self.stats["authors"][owner_name]['accepted'] += 1
            
            # Project
            self.stats["projects"][project]['total'] += 1
            if is_accepted: self.stats["projects"][project]['accepted'] += 1
            
            # Files
            for fp in file_list:
                self.stats["files"][fp]['total'] += 1
                if is_accepted: self.stats["files"][fp]['accepted'] += 1
            
            count += 1
            
        if count > 0:
            self.stats["last_updated"] = new_last_date
            self.save_to_disk()
            print(f"[Updater] Learned from {count} new changes.")


# --- 2. FEATURE EXTRACTOR (Matches FeatureEngineer Script) ---
class FeatureExtractor:
    def __init__(self):
        # Regex Patterns
        self.bug_id_pattern = re.compile(r"(?:Closes-Bug|Related-Bug|Bug):\s*#?(\d+)", re.IGNORECASE)
        self.security_pattern = re.compile(r"(CVE-\d+|Security|Vulnerability|Credential)", re.IGNORECASE)
        self.revert_pattern = re.compile(r"^Revert\s+\"", re.IGNORECASE)
        self.tag_pattern = re.compile(r"\[.*?\]") 
        self.bot_pattern = re.compile(r"\b(bot|zuul|jenkins|proposal)\b", re.IGNORECASE)
        
        # Categorization lists
        self.test_paths = ['test', 'tests', 'testing', 'zuul.d', '.zuul.yaml']
        self.dep_files = ['requirements.txt', 'test-requirements.txt', 'bindep.txt', 'setup.py']
        self.deploy_projects = {
            'openstack/kolla', 'openstack/kolla-ansible', 'openstack/kayobe', 
            'openstack/tripleo-heat-templates', 'openstack/puppet-openstack-integration',
            'openstack/openstack-ansible', 'openstack/bifrost'
        }

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

    def compute_features(self, change_data, history_mgr):
        """Converts Raw JSON -> Feature Dictionary for XGBoost"""
        
        # Basic parsing
        revision_info = change_data.get('revisions', {})
        latest_commit_hash = change_data.get('current_revision')
        if not revision_info or not latest_commit_hash: return None, None

        rev_data = revision_info[latest_commit_hash]
        commit = rev_data.get('commit', {})
        files = rev_data.get('files', {})
        msg = commit.get('message', "")
        subject = change_data.get('subject', "").lower()
        project_name = change_data.get('project', '')
        author_name = commit.get('author', {}).get('name', "Unknown")

        # --- 1. TEXT FEATURES ---
        feat_flesch_ease = textstat.flesch_reading_ease(msg)
        feat_gunning_fog = textstat.gunning_fog(msg)
        
        match_bug = self.bug_id_pattern.search(msg)
        feat_references_bug_tracker = 1 if match_bug else 0
        
        # --- 2. INTENT ---
        ci_keywords = ['ci', 'gate', 'pipeline', 'job', 'workflow', 'tox', 'lint', 'zuul', 'playbook']
        feat_is_ci_intent = 1 if any(k in subject for k in ci_keywords) else 0
        feat_is_feature = 1 if re.search(r'\b(add|implement|support|introduce|new|feat|enable|allow|provide)\b', subject) else 0
        feat_is_refactor = 1 if re.search(r'\b(refactor|clean|remove|move|rename|delete|drop)\b', subject) else 0
        feat_is_fix = 1 if re.search(r'\b(fix|resolve|repair|patch|correct|handle|mitigate|prevent)\b', subject) else 0
        feat_is_maintenance = 1 if re.search(r'\b(update|bump|upgrade|downgrade|pin|unpin|sync)\b', subject) else 0
        feat_is_deploy = 1 if re.search(r'\b(config|conf|deploy|install|set|use|default|variable|param|role)\b', subject) else 0

        # --- 3. FILE ANALYSIS ---
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

        feat_config_ratio = config_lines / total_lines if total_lines > 0 else 0.0
        feat_code_ratio = code_lines / total_lines if total_lines > 0 else 0.0
        feat_is_ci_change = 1 if (feat_is_ci_file_change == 1 or feat_is_ci_intent == 1) else 0
        feat_is_pure_config = 1 if (feat_config_ratio > 0.99) else 0

        # --- 4. ENTROPY & DENSITY ---
        feat_entropy = self.calculate_entropy(files)
        churn_raw = total_lines
        feat_churn_density = churn_raw / feat_file_count if feat_file_count > 0 else 0.0
        feat_churn_log = math.log(churn_raw + 1)
        total_deletions = sum(f.get('lines_deleted', 0) for f in files.values())
        feat_deletion_ratio = total_deletions / churn_raw if churn_raw > 0 else 0.0

        depths = [f.count('/') + 1 for f in file_list]
        feat_dir_depth = sum(depths) / len(depths) if depths else 0
        feat_ext_entropy = len(set([f.split('.')[-1] for f in file_list if '.' in f]))
        feat_is_deploy_project = 1 if any(dp in project_name for dp in self.deploy_projects) else 0
        feat_is_bot = 1 if self.bot_pattern.search(author_name) else 0

        # --- 5. HISTORICAL FEATURES (Time Travel Lookups) ---
        # Author Stats
        a_stats = history_mgr.stats["authors"].get(author_name, {'total': 0, 'accepted': 0, 'cumulative_churn': 0})
        feat_author_success = a_stats['accepted'] / a_stats['total'] if a_stats['total'] > 0 else 0.0
        feat_author_sub_count = a_stats['total']
        feat_author_trust = a_stats['total'] / (a_stats['cumulative_churn'] + 1)
        
        # Project Stats
        p_stats = history_mgr.stats["projects"].get(project_name, {'total': 0, 'accepted': 0})
        feat_proj_accept = p_stats['accepted'] / p_stats['total'] if p_stats['total'] > 0 else 0.0
        
        # File Probabilities
        f_probs = []
        for f in file_list:
            fs = history_mgr.stats["files"].get(f, {'total': 0, 'accepted': 0})
            f_probs.append(fs['accepted'] / fs['total'] if fs['total'] > 0 else 0.0)
        feat_hist_file_prob = max(f_probs) if f_probs else 0.0

        # --- 6. INTERACTIONS ---
        feat_safe_entropy = feat_entropy * feat_is_pure_config

        # --- ASSEMBLE DICT (Matching Model columns) ---
        features = {
            "safe_entropy_interaction": feat_safe_entropy,
            "is_bot": feat_is_bot,
            "is_deployment_project": feat_is_deploy_project,
            "is_pure_config": feat_is_pure_config,
            "is_fix": feat_is_fix,
            "is_feature": feat_is_feature,
            "is_maintenance": feat_is_maintenance,
            "is_deployment": feat_is_deploy,
            "is_ci_change": feat_is_ci_change,
            "is_refactor": feat_is_refactor,
            "config_line_ratio": feat_config_ratio,
            "code_line_ratio": feat_code_ratio,
            "churn_density": feat_churn_density,
            "change_entropy": feat_entropy,
            "file_count": feat_file_count,
            "churn_log_size": feat_churn_log,
            "deletion_ratio": feat_deletion_ratio,
            "msg_readability_ease": feat_flesch_ease,
            "msg_gunning_fog": feat_gunning_fog,
            "references_bug_tracker": feat_references_bug_tracker,
            "has_subject_tag": 1 if self.tag_pattern.search(subject) else 0,
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
            "has_gerrit_topic": 1 if change_data.get('topic') else 0,
            
            # Historical
            "author_success_rate": feat_author_success,
            "author_submission_count": feat_author_sub_count,
            "author_trust_score": feat_author_trust,
            "project_acceptance_rate": feat_proj_accept,
            "historical_file_prob": feat_hist_file_prob
        }
        
        # Determine Display Type for UI (for the LLM and the Frontend)
        display_type = "Feature"
        if feat_is_fix: display_type = "Bug Fix"
        elif feat_is_ci_change: display_type = "CI/Infra"
        elif feat_is_refactor: display_type = "Refactor"
        elif feat_modifies_deps: display_type = "Dependency"
        elif feat_is_maintenance: display_type = "Maintenance"
        elif features["is_documentation_only"]: display_type = "Docs"
        
        return features, display_type
# --- 3. LLM EXPLAINER (Generates AI Justifications) ---
class LLMExplainer:
    def __init__(self):
        try:
            self.client = Groq(api_key=Config.GROQ_API_KEY)
            self.model = "llama-3.3-70b-versatile"
        except: self.client = None

    def _format_vector(self, features):
        """
        Transforms the raw dictionary into a clean, readable JSON string for the AI.
        - Rounds floats to 3 decimals.
        - Converts Booleans to Yes/No.
        - Formats keys to Title Case.
        """
        clean_data = {}
        for key, value in features.items():
            # Make key readable (e.g. "author_trust_score" -> "Author Trust Score")
            readable_key = key.replace('_', ' ').title()
            
            if isinstance(value, bool) or value in [0, 1] and "is_" in key:
                clean_data[readable_key] = "Yes" if value else "No"
            elif isinstance(value, float):
                clean_data[readable_key] = round(value, 3)
            else:
                clean_data[readable_key] = value
        
        return json.dumps(clean_data, indent=2)

    def explain(self, msg, files, prob, features, display_type, threshold):
        if not self.client: return "AI Explanation unavailable."
        
        is_accepted = prob >= threshold
        verdict = "RECOMMENDED" if is_accepted else "NOT RECOMMENDED"
        
        # 1. Prepare File List
        file_list = list(files.keys())
        if len(file_list) > 50:
            file_str = "\n".join(file_list[:50]) + f"\n... (+ {len(file_list)-50} more)"
        else:
            file_str = "\n".join(file_list)

        # 2. Prepare Full Feature Vector (JSON)
        feature_vector_json = self._format_vector(features)

        # 3. Prompt with Context & Full Data
        prompt = f"""
        Act as a Senior OpenStack Release Manager. Justify the decision to {verdict} this backport.

        === DECISION ===
        VERDICT: {verdict} (Confidence: {prob:.1%}, Threshold: {threshold})
        CATEGORY: {display_type}

        === CONTEXT: HOW TO INTERPRET THE DATA ===
        - Author Trust Score: Ratio of accepted backports. 0.0=New, >0.5=Trusted.
        - Historical File Prob: Probability these specific files are usually backported.
        - Change Entropy: Code complexity (0=Simple, >4=Complex/Scattered).
        - Churn Density: Lines changed per file (High = Dense/Risky).
        - Modifies DB/API: Critical risk factors.

        === FULL FEATURE VECTOR (Internal Data) ===
        {feature_vector_json}

        === CHANGE ARTIFACTS ===
        Commit Message:
        "{msg}"

        Files Modified:
        {file_str}

        === INSTRUCTIONS ===
        Write a professional, 2-3 sentence justification.
        
        1. **Analyze the Vector:** Look at the "Full Feature Vector" above. Find the anomalies or strong signals.
        2. **Synthesize:** Do not list the numbers. Explain their *meaning*.
           - Instead of saying "Is Pure Config is Yes", say "The change is a low-risk configuration update."
           - Instead of saying "Trust is 0.0", say "The author lacks a prior track record."
        3. **Explain the Verdict:**
           - If REJECTED: Is it the Author? The Complexity? The specific Files? ...
           - If ACCEPTED: Is it the Safety (Config/Doc)? The High Trust? ...
        
        RESPONSE:
        """
        
        try:
            chat = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model, max_tokens=250, temperature=0.2
            )
            return chat.choices[0].message.content
        except Exception as e: return f"AI Error: {str(e)}"
# --- 4. FLASK APPLICATION ---
app = Flask(__name__)
CORS(app)

print("Starting Backport Assistant (Production Mode)...")

# Initialize Helpers
history = HistoryManager()
extractor = FeatureExtractor()
llm_explainer = LLMExplainer()

# Start Background Updater
bg_thread = threading.Thread(target=history.fetch_and_update, daemon=True)
bg_thread.start()

# Load XGBoost Model
model = None
try:
    model = xgb.Booster()
    model.load_model(Config.MODEL_PATH)
    print("XGBoost Model loaded successfully.")
except Exception as e:
    print(f"CRITICAL: Failed to load model from {Config.MODEL_PATH}. Error: {e}")

# Load User Threshold
try:
    with open(Config.THRESHOLD_PATH, "r") as f: THRESHOLD = float(f.read().strip())
except: THRESHOLD = 0.50

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data: return jsonify({"error": "No data provided"}), 400
        
        # 1. Calculate Features (includes History lookups)
        features, display_type = extractor.compute_features(data, history)
        
        if not features: 
            return jsonify({"error": "Invalid Change Data (Missing revisions or files)"}), 400

        # 2. Prepare DataFrame for XGBoost
        # Convert dict to DataFrame (automatically aligns columns if model has feature names)
        df = pd.DataFrame([features])
        df = df.astype(float)
        
        # 3. Predict
        dmatrix = xgb.DMatrix(df)
        prob = model.predict(dmatrix)[0]
        
        # 4. Generate AI Explanation
        raw_msg = data.get('revisions', {}).get(data.get('current_revision'), {}).get('commit', {}).get('message', '')
        raw_files = data.get('revisions', {}).get(data.get('current_revision'), {}).get('files', {})
        
        explanation = llm_explainer.explain(raw_msg, raw_files, prob, features, display_type, THRESHOLD)
        
        # 5. Build Response
        response = {
            "probability": float(prob),
            "ai_explanation": explanation,
            "features_used": {
                "nlp_type": display_type,
                "author_trust": float(features['author_trust_score']),
                "entropy": float(features['change_entropy']),
                "file_risk": float(features['historical_file_prob']),
                "churn_density": float(features['churn_density'])
            }
        }
        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/threshold', methods=['POST'])
def update_threshold():
    """Allows the user to adjust sensitivity from the Chrome Extension."""
    global THRESHOLD
    try:
        val = float(request.json.get('threshold'))
        if 0 <= val <= 1:
            THRESHOLD = val
            with open(Config.THRESHOLD_PATH, "w") as f: f.write(str(val))
            return jsonify({"status": "updated", "new_threshold": THRESHOLD})
    except: pass
    return jsonify({"error": "Invalid threshold"}), 400

if __name__ == '__main__':
    # Threaded=True allows handling multiple requests while background thread runs
    app.run(host='0.0.0.0', port=5000, threaded=True)