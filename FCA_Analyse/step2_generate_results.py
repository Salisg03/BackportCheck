import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df_raw = pd.read_csv('openstack_complete.csv')
df_features = pd.read_csv('fca_input_quantiles.csv')

df_features['Verdict_Accepted'] = (df_raw['target'] == 1).astype(int)
df_features['Verdict_Rejected'] = (df_raw['target'] == 0).astype(int)
data_bool = df_features.iloc[:, 1:].astype(bool)

frequent_itemsets = apriori(data_bool, min_support=0.02, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.65)

rules = rules[rules['lift'] >= 1.0]

csv_df = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
csv_df['antecedents'] = csv_df['antecedents'].apply(lambda x: list(x))
csv_df['consequents'] = csv_df['consequents'].apply(lambda x: list(x))
csv_df = csv_df.sort_values(by=['confidence', 'lift'], ascending=False)

csv_df.head(100).to_csv('top_100_rules.csv', index=False)
csv_df.to_csv('all_rules_report.csv', index=False)
print("CSV files generated: all_rules_report.csv and top_100_rules.csv")

def get_top_rules(category_type, limit=5):
    selected_rules = []
    sorted_rules = rules.sort_values(by=['confidence', 'lift', 'support'], ascending=False)
    
    for i, row in sorted_rules.iterrows():
        cons = list(row['consequents'])
        ants = list(row['antecedents'])
        
        is_accepted = 'Verdict_Accepted' in cons
        is_rejected = 'Verdict_Rejected' in cons
        is_config = 'Is_Config_Only' in ants or 'Is_Doc_Only' in ants
        
        if category_type == 'RISK' and is_rejected:
            selected_rules.append(row)
        elif category_type == 'SAFE_CONFIG' and is_accepted and is_config:
            selected_rules.append(row)
        elif category_type == 'SAFE_LOGIC' and is_accepted and not is_config:
            selected_rules.append(row)
            
        if len(selected_rules) >= limit:
            break
            
    return selected_rules

def clean_text(text_set):
    text = ", ".join(list(text_set))
    text = text.replace("Verdict_Accepted", "Accepted").replace("Verdict_Rejected", "Rejected")
    text = text.replace("_", "\\_")
    text = text.replace("Is\\_Config\\_Only", "Config")
    text = text.replace("msg\\_readability\\_ease", "Readability")
    text = text.replace("author\\_success\\_rate", "SuccessRate")
    text = text.replace("author\\_trust\\_score", "Trust")
    text = text.replace("directory\\_depth", "Depth")
    return text

print("\n\n=== COPIE CE CODE DANS OVERLEAF ===\n")
print(r"\begin{table*}[ht]")
print(r"\caption{Extracted Association Rules (Ordered by Confidence, Lift $\ge$ 1)}")
print(r"\label{tab:association_rules}")
print(r"\tiny")
print(r"\begin{tabular}{l l c c c}")
print(r"\toprule")
print(r"\textbf{Antecedents (Conditions)} & \textbf{Consequent} & \textbf{Supp.} & \textbf{Conf.} & \textbf{Lift} \\")
print(r"\midrule")

print(r"\multicolumn{5}{l}{\textit{\textbf{Risk Patterns (Rejection Signals)}}} \\")
for row in get_top_rules('RISK', 5):
    ant = clean_text(row['antecedents'])
    print(f"{ant} & Rejected & {row['support']:.2f} & {row['confidence']:.2f} & {row['lift']:.2f} \\\\")

print(r"\midrule")

print(r"\multicolumn{5}{l}{\textit{\textbf{Intrinsic Safety (Configuration/Docs)}}} \\")
for row in get_top_rules('SAFE_CONFIG', 5):
    ant = clean_text(row['antecedents'])
    print(f"{ant} & Accepted & {row['support']:.2f} & {row['confidence']:.2f} & {row['lift']:.2f} \\\\")

print(r"\midrule")

print(r"\multicolumn{5}{l}{\textit{\textbf{Logic Safety (Trusted Code Patterns)}}} \\")
for row in get_top_rules('SAFE_LOGIC', 6):
    ant = clean_text(row['antecedents'])
    print(f"{ant} & Accepted & {row['support']:.2f} & {row['confidence']:.2f} & {row['lift']:.2f} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table*}")