import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

img_file = "x_train_img.npz"
metadata_file = "x_train.csv"
y_file = "y_train.csv"

X_dev = pd.read_csv(metadata_file)
y_dev = pd.read_csv(y_file)
X_dev

# Function and code to load images
def load_img_data(file_path):
    with np.load(file_path) as data:
        img = data['images']
        ids = data['image_ids']
    print(f"Successfully loaded {img.shape[0]} images.")  
    return img, ids

imgs, img_ids = load_img_data(img_file)
print(img_ids.shape, imgs.shape)

# Make sure the IDs match and are in the same order
assert np.all(img_ids == X_dev['img_id']), "Image IDs in metadata and image data do not match"

# Show the first 5 images
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for i in range(5):
    axes[i].imshow(imgs[i])
    axes[i].axis('off')
    axes[i].set_title(f"Sample {i} ({y_dev.iloc[i]["fine_label"]})")

plt.show()

# These are the 3 columns in X that have categorical data.
# Everything else is binary or numeric. 
categorical_cols = ['background_father', 'background_mother', 'region'] 

# Get all obvious true/false cols that aren't already true/false
for col in X_dev.select_dtypes(include=['object']):
    if X_dev[col].dropna().astype(str).str.contains('True|False').any():
        print(f"Automatically converting {col} to binary...")
        unmapped = X_dev[~X_dev[col].isin(['True','False'])][col].unique()
        print("The following values are being mapped to NaN in this column: ", unmapped)
        X_dev[col] = X_dev[col].map({'True': 1, 'False': 0, True: 1, False: 0})

X_dev['gender'] = X_dev['gender'].map({'MALE': 0, 'FEMALE': 1})

# Handle the categorical data
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

encoded_feats = encoder.fit_transform(X_dev[categorical_cols])
encoded_df = pd.DataFrame(
    encoded_feats, 
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_dev.index
)

# Drop original text columns and join the new numeric ones
X_dev = X_dev.drop(columns=categorical_cols).join(encoded_df)

age_groups = pd.cut(
    X_dev['age'], 
    bins=[0, 30, 60, 100], 
    labels=['Young (<30)', 'Adult (30-60)', 'Senior (60+)']
)

# First we'll calculate how much missing data there is
X_dev.isna().sum()

# Then we'll decide what to do. For now we'll just treat missing data as false, 
# since it probably means that a patient didn't think the question was important or relevant.
X_dev = X_dev.fillna(0)
X_dev

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import GroupKFold
# from sklearn.metrics import roc_auc_score

# imgs_reshaped = imgs.reshape(imgs.shape[0], -1)

# img_features = pd.DataFrame({
#     'img_mean': imgs_reshaped.mean(axis=1),
#     'img_std':  imgs_reshaped.std(axis=1),
# })

# data = X_dev.copy()
# data['coarse_label'] = y_dev['coarse_label']
# data = pd.concat([data, img_features], axis=1)

# X      = data.drop(columns=['coarse_label', 'img_id', 'patient_id'])
# y      = data['coarse_label']
# groups = data['patient_id']

# param_grid = {
#     'max_depth':         [3, 5, 10, 15, 20, None],
#     'n_estimators':     [50, 100, 200],
#     'min_samples_leaf': [1, 2, 5],
# }

# gkf = GroupKFold(n_splits=5)
# folds = list(gkf.split(X, y, groups))

# best_auc = -1
# best_params = None
# all_results = []

# for max_depth in param_grid['max_depth']:
#     for n_est in param_grid['n_estimators']:
#         for msl in param_grid['min_samples_leaf']:
#             fold_train_aucs = []
#             fold_val_aucs   = []

#             for train_idx, val_idx in folds:
#                 X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
#                 y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

#                 clf = RandomForestClassifier(
#                     n_estimators=n_est,
#                     max_depth=max_depth,
#                     min_samples_leaf=msl,
#                     class_weight='balanced',
#                     random_state=42,
#                     n_jobs=-1,
#                 )
#                 clf.fit(X_tr, y_tr)

#                 fold_train_aucs.append(roc_auc_score(y_tr, clf.predict_proba(X_tr)[:, 1]))
#                 fold_val_aucs.append(roc_auc_score(y_vl, clf.predict_proba(X_vl)[:, 1]))

#             mean_val = np.mean(fold_val_aucs)
#             all_results.append({
#                 'max_depth':        max_depth,
#                 'n_estimators':    n_est,
#                 'min_samples_leaf': msl,
#                 'mean_val_auc':    mean_val,
#                 'mean_train_auc':  np.mean(fold_train_aucs),
#                 'fold_val_aucs':   fold_val_aucs,
#                 'fold_train_aucs': fold_train_aucs,
#             })

#             if mean_val > best_auc:
#                 best_auc    = mean_val
#                 best_params = {
#                     'max_depth':        max_depth,
#                     'n_estimators':    n_est,
#                     'min_samples_leaf': msl,
#                 }

# print(f"Best mean val AUROC: {best_auc:.4f}")
# print(f"Best params: {best_params}")

# best_n_est = best_params['n_estimators']
# best_msl   = best_params['min_samples_leaf']

# plot_rows = [r for r in all_results
#              if r['n_estimators'] == best_n_est
#              and r['min_samples_leaf'] == best_msl]
# plot_rows.sort(key=lambda r: (r['max_depth'] is None, r['max_depth'] or 0))

# x_labels    = [str(r['max_depth']) if r['max_depth'] is not None else 'None\n(unlimited)' for r in plot_rows]
# x_pos       = np.arange(len(plot_rows))
# mean_val_aucs   = [r['mean_val_auc']   for r in plot_rows]
# mean_train_aucs = [r['mean_train_auc'] for r in plot_rows]

# fig, ax = plt.subplots(figsize=(8, 5))

# ax.plot(x_pos, mean_val_aucs,   'o-',  color='#185FA5', lw=2, label='Validation AUROC (mean)', zorder=3)
# ax.plot(x_pos, mean_train_aucs, 's--', color='#D85A30', lw=2, label='Train AUROC (mean)',      zorder=3)

# for i, r in enumerate(plot_rows):
#     for v in r['fold_val_aucs']:
#         ax.scatter(i, v, color='#185FA5', alpha=0.35, s=30, zorder=2)
#     for v in r['fold_train_aucs']:
#         ax.scatter(i, v, color='#D85A30', alpha=0.35, s=30, marker='s', zorder=2)

# ax.set_xticks(x_pos)
# ax.set_xticklabels(x_labels)
# ax.set_xlabel('max_depth', fontsize=12)
# ax.set_ylabel('AUROC', fontsize=12)
# ax.set_ylim([0.5, 1.01])
# ax.legend(fontsize=10)
# ax.grid(axis='y', alpha=0.3)
# ax.set_title(
#     f'AUROC vs max_depth (n_estimators={best_n_est}, min_samples_leaf={best_msl})',
#     fontsize=11
# )
# fig.text(0.5, -0.04,
#     f'Figure 1. Validation and training AUROC across max_depth values with n_estimators={best_n_est} and min_samples_leaf={best_msl} fixed to best values. Each faint dot is one CV fold; solid lines show the mean.',
#     ha='center', fontsize=9, color='gray', wrap=True)

# plt.tight_layout()
# plt.savefig('fig1_max_depth.png', dpi=150, bbox_inches='tight')
# plt.show()

# final_model = RandomForestClassifier(
#     n_estimators=best_params['n_estimators'],
#     max_depth=best_params['max_depth'],
#     min_samples_leaf=best_params['min_samples_leaf'],
#     class_weight='balanced',
#     random_state=42,
#     n_jobs=-1,
# )
# final_model.fit(X, y)

# val_results = []

# for train_idx, val_idx in folds:
#     X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
#     y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

#     fold_model = RandomForestClassifier(
#         n_estimators=best_params['n_estimators'],
#         max_depth=best_params['max_depth'],
#         min_samples_leaf=best_params['min_samples_leaf'],
#         class_weight='balanced',
#         random_state=42,
#         n_jobs=-1,
#     )
#     fold_model.fit(X_tr, y_tr)
#     proba = fold_model.predict_proba(X_vl)[:, 1]

#     for idx, p, true_label in zip(val_idx, proba, y_vl.values):
#         val_results.append({
#             'orig_idx':   idx,
#             'proba':      p,
#             'true_label': true_label,
#             'predicted':  1 if p >= 0.5 else 0,
#         })

# val_df = pd.DataFrame(val_results)
# val_df['correct'] = val_df['predicted'] == val_df['true_label']
# val_df['confidence'] = val_df['proba'].apply(lambda p: max(p, 1 - p))

# correct_df   = val_df[val_df['correct']].nlargest(3, 'confidence')
# incorrect_df = val_df[~val_df['correct']].nlargest(3, 'confidence')

# full_data = data.reset_index(drop=True)

# print("=== 3 Most Confidently CORRECT ===")
# for _, row in correct_df.iterrows():
#     i = row['orig_idx']
#     meta = full_data.iloc[i]
#     print(f"  idx={i} | proba={row['proba']:.4f} | true={row['true_label']} | age={meta['age']}")

# print()
# print("=== 3 Most Confidently INCORRECT ===")
# for _, row in incorrect_df.iterrows():
#     i = row['orig_idx']
#     meta = full_data.iloc[i]
#     print(f"  idx={i} | proba={row['proba']:.4f} | true={row['true_label']} | age={meta['age']}")
    
# fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# for ax, (_, row) in zip(axes[0], correct_df.iterrows()):
#     i = row['orig_idx']
#     ax.imshow(imgs[i])
#     ax.axis('off')
#     ax.set_title(
#         f"CORRECT\ntrue={row['true_label']} | p={row['proba']:.3f}\nage={full_data.iloc[i]['age']:.0f}",
#         fontsize=9
#     )

# for ax, (_, row) in zip(axes[1], incorrect_df.iterrows()):
#     i = row['orig_idx']
#     ax.imshow(imgs[i])
#     ax.axis('off')
#     ax.set_title(
#         f"INCORRECT\ntrue={row['true_label']} | p={row['proba']:.3f}\nage={full_data.iloc[i]['age']:.0f}",
#         fontsize=9
#     )

# fig.suptitle('Problem 1: Most Confidently Correct (top) and Incorrect (bottom) Predictions', fontsize=11)
# plt.tight_layout()
# plt.savefig('fig1_predictions.png', dpi=150, bbox_inches='tight')
# plt.show()


# X_test = pd.read_csv("x_test.csv")

# for col in X_test.select_dtypes(include=['object']):
#     if X_test[col].dropna().astype(str).str.contains('True|False').any():
#         X_test[col] = X_test[col].map({'True': 1, 'False': 0, True: 1, False: 0})

# X_test['gender'] = X_test['gender'].map({'MALE': 0, 'FEMALE': 1})

# encoded_feats_test = encoder.transform(X_test[categorical_cols])
# encoded_df_test = pd.DataFrame(
#     encoded_feats_test,
#     columns=encoder.get_feature_names_out(categorical_cols),
#     index=X_test.index
# )

# X_test = X_test.drop(columns=categorical_cols).join(encoded_df_test)
# X_test = X_test.fillna(0)

# test_imgs, test_img_ids = load_img_data("x_test_img.npz")
# test_imgs_reshaped = test_imgs.reshape(test_imgs.shape[0], -1)

# img_features_test = pd.DataFrame({
#     'img_mean': test_imgs_reshaped.mean(axis=1),
#     'img_std':  test_imgs_reshaped.std(axis=1),
# })

# X_test = pd.concat([X_test, img_features_test], axis=1)
# X_test_final = X_test.drop(columns=['img_id', 'patient_id'])

# y_test_proba = final_model.predict_proba(X_test_final)[:, 1]
# np.savetxt("yproba1_test.txt", y_test_proba)