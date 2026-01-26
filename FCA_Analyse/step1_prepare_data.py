import pandas as pd
import numpy as np

df = pd.read_csv('openstack_complete.csv')

fca_df = pd.DataFrame()
fca_df['id'] = df.iloc[:, 0]

numeric_features = [
    'change_entropy', 'churn_density', 'file_count', 'directory_depth',
    'code_line_ratio', 'config_line_ratio', 'churn_log_size',
    'author_trust_score', 'author_submission_count', 'author_success_rate',
    'historical_file_prob', 'msg_readability_ease', 'file_extension_entropy'
]

for col in numeric_features:
    if col not in df.columns:
        continue
    
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    
    if q1 == q3:
        fca_df[f'{col}_High'] = (df[col] > q3).astype(int)
        fca_df[f'{col}_Low']  = (df[col] <= q3).astype(int)
    else:
        fca_df[f'{col}_Low'] = (df[col] <= q1).astype(int)
        fca_df[f'{col}_Med'] = ((df[col] > q1) & (df[col] <= q3)).astype(int)
        fca_df[f'{col}_High'] = (df[col] > q3).astype(int)

bool_map = {
    'is_bot': 'Is_Bot',
    'is_pure_config': 'Is_Config_Only',
    'is_fix': 'Is_BugFix',
    'is_feature': 'Is_Feature',
    'is_refactor': 'Is_Refactor',
    'is_documentation_only': 'Is_Doc_Only',
    'is_maintenance': 'Is_Maintenance',
    'modifies_db_migration': 'Modifies_DB',
    'modifies_public_api': 'Modifies_API',
    'modifies_dependencies': 'Modifies_Deps',
    'has_security_impact': 'Security_Fix',
    'is_test_change': 'Is_Test',
    'is_ci_change': 'Is_CI'
}

for old_col, new_name in bool_map.items():
    if old_col in df.columns:
        fca_df[new_name] = df[old_col].astype(int)

output_file = 'fca_input_quantiles.csv'
fca_df.to_csv(output_file, index=False)
print(f"File generated: {output_file}")