import os
import json
import time
import threading
import re
import math
import requests
import pandas as pd
import xgboost as xgb
import pickle 
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()


class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "xgboost_optimized.json")
    STATS_PATH = os.path.join(BASE_DIR, "stats_complete.json")
    THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.txt")
    PCA_PATH = os.path.join(BASE_DIR, "pca_model.pkl")
    
    GERRIT_API = "https://review.opendev.org"
    UPDATE_INTERVAL = 3600
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    REGEX_BUG = re.compile(r"(?:closes-bug|bug|related-bug):\s*#?(\d+)", re.IGNORECASE)
    REGEX_CVE = re.compile(r"(cve-\d{4}-\d+|security|credential|vulnerability)", re.IGNORECASE)
    REGEX_REVERT = re.compile(r"^revert", re.IGNORECASE)
    REGEX_SUBJECT_TAG = re.compile(r"^\[.*?\]")

# 2. GESTIONNAIRE D'HISTORIQUE 

class HistoryManager:
    def __init__(self):
        self.stats = {
            "authors": defaultdict(lambda: {'submissions': 0, 'accepted_backports': 0, 'total_churn': 0}),
            "files": defaultdict(lambda: {'touched': 0, 'backported': 0}),
            "projects": defaultdict(lambda: {'submissions': 0, 'accepted_backports': 0}),
            "last_updated": "2020-01-01 00:00:00"
        }
        self.load_from_disk()
    
    def load_from_disk(self):
        """
        Loads historical statistics from the 'stats_complete.json' file into memory.
        This allows the system to persist knowledge about author trust and file risks across server restarts.
        """
        if os.path.exists(Config.STATS_PATH):
            try:
                with open(Config.STATS_PATH, "r") as f:
                    data = json.load(f)
                    self.stats["authors"].update(data.get("authors", {}))
                    self.stats["files"].update(data.get("files", {}))
                    self.stats["projects"].update(data.get("projects", {}))
                    self.stats["last_updated"] = data.get("meta_last_updated", self.stats["last_updated"])
                print(f"[History] Stats chargées.")
            except Exception as e: print(f"⚠ [History] Erreur lecture: {e}")
    #
    def save_to_disk(self):
        """
        Persists the current in-memory statistics to the local JSON file.
        This ensures that learning acquired from the live Gerrit stream is saved permanently.
        """
        snapshot = {
            "meta_last_updated": self.stats["last_updated"],
            "authors": dict(self.stats["authors"]),
            "files": dict(self.stats["files"]),
            "projects": dict(self.stats["projects"])
        }
        with open(Config.STATS_PATH, "w") as f: json.dump(snapshot, f)

    def fetch_and_update(self):
        """
        Background task that periodically polls the Gerrit API for new changes.
        It retrieves changes closed after the last recorded timestamp to incrementally update the statistics.
        """
        while True:
            try:
                # CLEANUP: Remove nanoseconds from the date for Gerrit compatibility
                raw_date = self.stats["last_updated"]
                clean_date = raw_date.split('.')[0] 
                
                print(f"[Updater] Scan since {clean_date}...")
                
                # Strict filter for training
                vote_filter = "(label:Backport-Candidate=-2 OR label:Backport-Candidate=-1 OR label:Backport-Candidate=+1 OR label:Backport-Candidate=+2)"
                query = f'status:closed after:"{clean_date}" AND {vote_filter}'
                
                resp = requests.get(
                    f"{Config.GERRIT_API}/changes/",
                    params={'q': query, 'o': ['CURRENT_REVISION', 'CURRENT_FILES', 'DETAILED_LABELS', 'DETAILED_ACCOUNTS']},
                    timeout=20
                )
                
                if resp.status_code == 200:
                    text = resp.text[4:] if resp.text.startswith(")]}'") else resp.text
                    changes = json.loads(text)
                    if changes:
                        self._process_changes(changes)
                    else:
                        print("[Updater] Nothing new.")
                else:
                    print(f"[Updater] Error Gerrit API: {resp.status_code}")
            except Exception as e:
                print(f"[Updater] Error: {e}")
            time.sleep(Config.UPDATE_INTERVAL)

    def _process_changes(self, changes):
        """
        Analyzes a batch of fetched changes to update trust metrics.
        It parses the 'Backport-Candidate' labels to determine if a past submission was accepted (positive sample) or rejected, updating author and file statistics accordingly.
        """
        count = 0
        new_last_date = self.stats["last_updated"]
        
        for change in changes:
            created = change.get('updated', change.get('created'))
            if created > new_last_date: new_last_date = created
            
            # Analyze Target 
            # Extract and sort votes by date to find the final decision
            votes = change.get('labels', {}).get('Backport-Candidate', {}).get('all', [])
            valid_votes = [v for v in votes if 'date' in v and 'value' in v]
            if not valid_votes: continue
            valid_votes.sort(key=lambda x: x['date'])
            final_score = int(valid_votes[-1]['value'])
            
            if final_score == 0: continue
            success = 1 if final_score > 0 else 0
            
            owner = str(change.get('owner', {}).get('_account_id', 'unknown'))
            project = change.get('project', 'unknown')
            rev = change.get('current_revision')
            if not rev: continue
            files = change['revisions'][rev].get('files', {})
            
            churn = sum(m.get('lines_inserted',0) + m.get('lines_deleted',0) for f,m in files.items() if f != "/COMMIT_MSG")
            
            self.stats["authors"][owner]['submissions'] += 1
            self.stats["authors"][owner]['total_churn'] += churn
            if success: self.stats["authors"][owner]['accepted_backports'] += 1
            
            self.stats["projects"][project]['submissions'] += 1
            if success: self.stats["projects"][project]['accepted_backports'] += 1
            
            for fp in files:
                if fp == "/COMMIT_MSG": continue
                self.stats["files"][fp]['touched'] += 1
                if success: self.stats["files"][fp]['backported'] += 1
            count += 1
            
        if count > 0:
            self.stats["last_updated"] = new_last_date
            self.save_to_disk()
            print(f"[Updater] {count} changes learned.")


# 3. SEMANTIC

class SemanticEngine:
    def __init__(self):
        """
        Initializes the Deep Learning models.
        Loads the SentenceTransformer (CodeBERT) for embedding generation and the pre-trained PCA model for dimensionality reduction.
        """
        self.bert = None
        self.pca = None
        try:
            print("Loading CodeBERT & PCA...")

            model_path = os.path.join(Config.BASE_DIR, 'model_cache')
            if os.path.exists(model_path):
                print(f"Loading locally from {model_path}...")
                self.bert = SentenceTransformer(model_path)
            else:
                # Fallback to internet (in case the folder doesn't exist)
                self.bert = SentenceTransformer('microsoft/codebert-base')

            with open(Config.PCA_PATH, "rb") as f:
                self.pca = pickle.load(f)
            print("Semantic Engine loaded.")
        except Exception as e: print(f"Semantic Error: {e}")

    def get_features(self, message):
        """
        Converts a raw commit message into a compressed semantic vector.
        It cleans the message, generates a 768-dimensional embedding via CodeBERT, and reduces it to 15 dimensions using PCA.
        """
        if not self.bert or not self.pca: return [0.0] * 15
        clean_lines = [line for line in message.split('\n') if not re.match(r'^(Change-Id|Signed-off-by):', line)]
        embedding = self.bert.encode(["\n".join(clean_lines).strip()])
        return self.pca.transform(embedding)[0]


# 4. FEATURE ENGINE
class FeatureEngine:
    @staticmethod
    def get_extension(filename):
        """
        Utility function to extract the file extension from a given file path.
        Used to analyze file types (e.g., documentation vs. configuration).
        """
        return '.' + filename.split('.')[-1].lower() if '.' in filename and not filename.startswith('/') else 'none'
    
    @staticmethod
    def count_syllables(word):
        """
        Heuristic function to count syllables in a word based on vowel patterns.
        Used as a primitive for calculating readability scores (Flesch/Gunning Fog).
        """
        word = word.lower(); count = 0; vowels = "aeiouy"
        if not word: return 0
        if word[0] in vowels: count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels: count += 1
        if word.endswith("e"): count -= 1
        if count == 0: count += 1
        return count

    @staticmethod
    def analyze_text_metrics(message):
        """
        Computes linguistic complexity metrics for the commit message.
        Returns average sentence length, Flesch Reading Ease score, and Gunning Fog index to evaluate message clarity."""
        tokens = message.split(); token_count = len(tokens)
        sentences = [s for s in re.split(r'[.!?]+', message) if s.strip()]
        avg_len = token_count / max(len(sentences), 1)
        syll = sum(FeatureEngine.count_syllables(t) for t in tokens)
        complex_w = sum(1 for t in tokens if FeatureEngine.count_syllables(t) >= 3)
        if token_count > 0:
            flesch = 206.835 - (1.015 * avg_len) - (84.6 * (syll / token_count))
            fog = 0.4 * (avg_len + 100 * (complex_w / token_count))
        else: flesch=0; fog=0
        return avg_len, flesch, fog

    @staticmethod
    def smart_classify_display(msg, paths):
        """
        UI Classification: Refactor, Revert, Dependency Upgrade
        Prioritizes certain keywords to assign a more specific display type for better UX.
        """
        msg_l = msg.lower(); subject = msg_l.split('\n')[0]
        
        # 1. REFACTOR / STYLE / LINT 
        # ex: "ansible-lint:", "pep8:", "flake8", "lint fix", "whitespace"
        if re.search(r'\b(lint(ing|er)?|pep8|flake8|hacking|style|formatting|whitespace|indentation)\b', subject):
            return "Refactor"
        
        # Detect classic terms: refactor, cleanup, dead code
        if re.search(r'\b(refactor(ing|ed)?|clean[- ]?up|prune|dead[- ]?code|unused)\b', subject):
            return "Refactor"
        
        # Explicit tags
        if re.match(r'^(chore|style|nit|build|refactor)', subject):
            return "Refactor"
            
        # 2. REVERT
        if re.match(r'^revert\b', subject) or "this reverts commit" in msg_l:
            return "Revert"

        # 3. DEPENDENCY
        if re.search(r'\b(bump|upgrade|update)\b.*\b(dependency|requirements|version|lib|tox)\b', subject):
            return "Dependency Upgrade"

        return None

    @staticmethod
    def _get_strict_model_type(msg, paths):
        """
        Classification Modèle
        """
        msg_l = msg.lower()
        subject = msg_l.split('\n')[0].strip()
        
        #1.BUG FIX (Priority Detection)**
        #A.The word "Fix" at the very beginning (e.g., "Fix idempotence...")
        #Accepts "fix" followed by a space, or "fixes", "fixed"
        if re.match(r'^fix(es|ed|ing)?\s+', subject):
            return "Bug Fix"
            
        # ex: "issue with podman fixed", "bug in nova", "patch for crash"
        strong_keywords = r'\b(hot[-]?fix|bug[-]?fix|quick[-]?fix|patch|crash|panic|traceback|exception|error|failure)\b'
        if re.search(strong_keywords, subject):
            return "Bug Fix"

        #C. Analysis of the message body (Official OpenStack tags)
        # ex: "Closes-Bug: #123", "Closes-Bug: 123", "Related-Bug:", "Partial-Bug:"
        if re.search(r'^\s*(closes|related|partial)-bug:\s*#?\d+', msg_l, re.MULTILINE):
            return "Bug Fix"
        # 2. DOC
        if re.search(r'\b(doc|docs|documentation|readme|typo|guide)\b', subject): return "Doc"
        if paths and all(FeatureEngine.get_extension(f) in ['.rst', '.md', '.txt', '.png', '.svg'] for f in paths): return "Doc"
        # 3. CI
        if re.search(r'\b(ci|test(s|ing)?|unit[-]?test|coverage|gate|zuul|tox)\b', subject): return "CI"
        if paths and all(('test' in f or 'zuul.d' in f or '.tox' in f) for f in paths): return "CI"

        # 4. FEATURE 
        return "Feature"
    @staticmethod
    def get_launchpad_metrics(message):
        """
        Interroge l'API Launchpad si un ID de bug est trouvé.
        Retourne (heat, severity_score, comments).
        """
        match = Config.REGEX_BUG.search(message)
        if match:
            bug_id = match.group(1)
            try:
                # Max 2s timeout to avoid slowing down the app
                r = requests.get(f"https://api.launchpad.net/1.0/bugs/{bug_id}", timeout=2)
                if r.status_code == 200:
                    d = r.json()
                    
                    # Mapping textual importance to an integer (like in training)
                    sev_map = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1, 'Wishlist': 0, 'Undecided': 0}
                    severity = sev_map.get(d.get('importance', 'Undecided'), 0)
                    
                    return int(d.get('heat', 0)), severity, int(d.get('message_count', 0))
            except Exception as e:
                print(f"⚠ Launchpad API Warning: {e}")
                pass
        return 0, 0, 0

    @staticmethod
    def build_vector(data, history, sem_engine):
        rev = list(data.get('revisions', {}).values())[0]
        files = rev.get('files', {})
        msg = rev.get('commit', {}).get('message', '')
        subject = data.get('subject', '')
        owner = str(data.get('owner', {}).get('_account_id', 'unknown'))
        project = data.get('project', 'unknown')

        sem_vec = sem_engine.get_features(msg)
        paths = [f for f in files if f != "/COMMIT_MSG"]
        
        # Classification
        nlp_type = FeatureEngine._get_strict_model_type(msg, paths)
        display_type = FeatureEngine.smart_classify_display(msg, paths) or nlp_type

        # Stats Churn
        churns = [m.get('lines_inserted',0)+m.get('lines_deleted',0) for f,m in files.items() if f != "/COMMIT_MSG"]
        total_churn = sum(churns)
        deletions = sum(m.get('lines_deleted',0) for f,m in files.items() if f != "/COMMIT_MSG")

        a_stat = history.stats["authors"].get(owner, {'submissions': 0, 'accepted_backports': 0, 'total_churn': 0})
        p_stat = history.stats["projects"].get(project, {'submissions': 0, 'accepted_backports': 0})
        avg_len, flesch, fog = FeatureEngine.analyze_text_metrics(msg)
        heat, sev, comments = FeatureEngine.get_launchpad_metrics(msg)
        f = {}
        f['1_references_bug_tracker'] = 1 if Config.REGEX_BUG.search(msg) else 0
        sub = a_stat['submissions']
        f['2_author_success_rate'] = a_stat['accepted_backports'] / sub if sub > 0 else 0
        f['3_churn_log_size'] = math.log(total_churn + 1)
        f['5_deletion_ratio'] = deletions / total_churn if total_churn > 0 else 0
        f['6_file_count'] = len(paths)
        f['8_avg_sentence_length'] = avg_len
        f['9_msg_complexity'] = len(msg)
        f['10_has_security_impact'] = 1 if Config.REGEX_CVE.search(msg) else 0
        f['11_nlp_change_type'] = nlp_type
        
        is_test = 1 if all(("test" in p.lower() or "zuul" in p or ".tox" in p) for p in paths) and paths else 0
        f['12_is_test_change'] = is_test
        f['13_is_revert'] = 1 if Config.REGEX_REVERT.match(subject) else 0
        f['14_modifies_dependencies'] = 1 if any("requirements.txt" in p or "bindep.txt" in p for p in paths) else 0
        f['15_author_submission_count'] = sub
        f['16_author_trust_score'] = sub / (a_stat['total_churn'] + 1)
        
        f_probs = []
        for p in paths:
            fs = history.stats["files"].get(p, {'touched': 0, 'backported': 0})
            if fs['touched'] > 0: f_probs.append(fs['backported'] / fs['touched'])
            else: f_probs.append(0.0)
        f['17_historical_file_prob'] = max(f_probs) if f_probs else 0.0
        f['18_is_documentation_only'] = 1 if all(FeatureEngine.get_extension(p) in ['.rst','.md','.txt'] for p in paths) and paths else 0
        f['19_author_file_confidence'] = f['2_author_success_rate'] * f['17_historical_file_prob']
        depths = [p.count('/') for p in paths]
        f['20_risk_module_coupling'] = max(depths) if depths else 0
        f['21_file_extension_entropy'] = len(set(FeatureEngine.get_extension(p) for p in paths))
        f['22_relation_depth'] = len(rev.get('commit', {}).get('parents', [])) - 1
        p_sub = p_stat['submissions']
        f['23_project_acceptance_rate'] = p_stat['accepted_backports'] / p_sub if p_sub > 0 else 0
        
        entropy = 0.0
        if total_churn > 0:
            for c in churns:
                if c > 0: p = c / total_churn; entropy -= p * math.log2(p)
        f['24_change_entropy'] = entropy
        f['25_directory_depth'] = sum(depths)/len(depths) if depths else 0
        f['26_msg_readability_ease'] = flesch
        f['27_msg_gunning_fog'] = fog
        f['28_has_gerrit_topic'] = 1 if data.get('topic') else 0
        f['29_has_subject_tag'] = 1 if Config.REGEX_SUBJECT_TAG.match(subject) else 0
        
        test_churn_val = sum(c for i, c in enumerate(churns) if "test" in paths[i].lower() or "zuul" in paths[i])
        f['31_test_code_ratio'] = test_churn_val / total_churn if total_churn > 0 else 0
        
        f['32_config_change'] = 1 if any(FeatureEngine.get_extension(p) in ['.conf','.ini','.yaml','.json'] for p in paths) else 0
        f['33_desc_density'] = len(msg) / (total_churn + 1)
        
        is_weekend = 0
        try:
            dt = datetime.strptime(data.get('created','').split('.')[0], "%Y-%m-%d %H:%M:%S")
            if dt.weekday() >= 5: is_weekend = 1
        except: pass
        f['34_is_weekend'] = is_weekend
        f['35_bug_heat'] = heat
        f['36_bug_severity'] = sev
        f['37_bug_comments'] = comments
        f['38_is_bot'] = 1 if "bot" in data.get('owner', {}).get('name', '').lower() else 0

        for i, v in enumerate(sem_vec): f[f'40_sem_vec_{i}'] = v
        f['full_text_dl'] = ""
        return f, display_type
# 5. LLM EXPLAINER

class LLMExplainer:
    def __init__(self):
        """
        Initializes the client for the Groq API.
        Sets up the connection to use the Llama-3 model for natural language generation.
        """
        try:
            self.client = Groq(api_key=Config.GROQ_API_KEY)
            self.model = "llama-3.3-70b-versatile"
        except: self.client = None

    def _format_features_for_ai(self, features):
        """Cleans up technical feature names into readable text for the AI."""
        readable_map = {}
        for key, value in features.items():
            if "sem_vec" in key or "full_text" in key: continue
            clean_key = re.sub(r'^\d+_', '', key).replace('_', ' ').title()
            if isinstance(value, float): readable_map[clean_key] = round(value, 4)
            else: readable_map[clean_key] = value
        return json.dumps(readable_map, indent=2)

    def explain(self, msg, files, prob, features, display_type, threshold):
        """
        Generates a natural language justification for the prediction.
        Constructs a prompt containing the decision context, metrics, and file list, then queries the LLM to explain why the change was accepted or rejected.
        """
        if not self.client: return "AI unavailable."
        
        is_accepted = prob >= threshold
        verdict = "RECOMMENDED (Backport Candidate)" if is_accepted else "NOT RECOMMENDED (Rejected)"
        
        all_files = list(files.keys())
        files_text = "\n".join([f"- {f}" for f in all_files])
        
        # Safety net: If the file list is truly massive (> 4000 chars), we truncate to avoid API errors
        if len(files_text) > 4000:
            files_text = files_text[:4000] + "\n... (List truncated: too many files)"

        # 2. Full Commit Message (2000 chars is usually enough for even very long descriptions)
        full_msg = msg[:2000] 

        # Dynamic Instructions (To enforce alignment with the XGBoost score)
        if is_accepted:
            instruction = "Explain why this change is safe to backport. Focus on High Trust, Low Risk/Impact, or the critical nature of the fix."
        else:
            instruction = "Explain the rejection. Focus on risks like Low Author Trust, High Code Complexity (Entropy), or the fact it looks like a Feature/Refactor."
            
        prompt = f"""
        Act as a Senior OpenStack Release Manager. 
        Evaluate this commit based on the predictive risk profile below.
        
        === DECISION CONTEXT ===
        VERDICT: {verdict}
        CONFIDENCE: {prob:.1%} (Threshold: {threshold})
        TYPE: {display_type}

        === RISK METRICS (Internal Data) ===
        {self._format_features_for_ai(features)}

        === COMMIT CONTENT ===
        Message: 
        {full_msg}
        
        Files Modified ({len(all_files)} total):
        {files_text}

        === INSTRUCTION ===
        {instruction}
        

        Guidelines:
        1. **Prioritize qualitative reasoning over listing raw numbers.** You may cite a specific metric (e.g. "Trust Score") ONLY if it is the main reason for the decision, but avoid simply reading out the stats.
        2. **Connect the dots**: Relate the file types or commit message to the risk score.
        3. **Interpret the metrics**
        4. Be professional, direct, and concise (2 sentences max).
        """
        try:
            chat = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model, 
                max_tokens=150,
                temperature=0.1
            )
            return chat.choices[0].message.content
        except Exception as e: 
            return f"AI Analysis failed: {str(e)}"

# 6. APP MAIN

app = Flask(__name__)
CORS(app)

print("Backport Assistant (Full AI)")
history = HistoryManager()
sem_engine = SemanticEngine()
llm = LLMExplainer()
bg_thread = threading.Thread(target=history.fetch_and_update, daemon=True)
bg_thread.start()

try:
    model = xgb.Booster()
    model.load_model(Config.MODEL_PATH)
    print("XGBoost loaded.")
except: print("XGBoost Error.")

try: 
    with open(Config.THRESHOLD_PATH, "r") as f: THRESHOLD = float(f.read().strip())
except: THRESHOLD = 0.50

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main API Endpoint.
    1. Receives the Change JSON from the Chrome Extension.
    2. Calls FeatureEngine to build the mathematical vector.
    3. Runs the XGBoost model to predict the backport probability.
    4. Calls the LLM to generate a text explanation.
    5. Returns a JSON response containing the probability, explanation, and key metrics.
    """
    try:
        data = request.json
        features, display_type = FeatureEngine.build_vector(data, history, sem_engine)
        
        df = pd.DataFrame([features])
        t = features['11_nlp_change_type']
        df['11_nlp_change_type_CI'] = 1 if t == 'CI' else 0
        df['11_nlp_change_type_Doc'] = 1 if t == 'Doc' else 0
        df['11_nlp_change_type_Feature'] = 1 if t == 'Feature' else 0
        
        cols_drop = ['11_nlp_change_type', 'full_text_dl']
        X = df.drop(columns=cols_drop).astype(float)
        
        if hasattr(model, 'feature_names'):
            for c in model.feature_names:
                if c not in X.columns: X[c] = 0.0
            X = X[model.feature_names]
            
        prob = model.predict(xgb.DMatrix(X))[0]
        
        raw_msg = data.get('revisions', {}).get(data.get('current_revision'), {}).get('commit', {}).get('message', '')
        raw_files = data.get('revisions', {}).get(data.get('current_revision'), {}).get('files', {})
        
        ai_text = llm.explain(raw_msg, raw_files, prob, features, display_type, THRESHOLD)
        
        return jsonify({
            "probability": float(prob),
            "ai_explanation": ai_text,
            "features_used": {
                "nlp_type": display_type,
                "max_path_depth": int(features['20_risk_module_coupling']),
                "author_trust": float(features['16_author_trust_score']),
                "entropy": float(features['24_change_entropy']),
                "launchpad_heat": int(features['35_bug_heat'])
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # host='0.0.0.0' is required for Docker
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)