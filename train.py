import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
os.environ["OMP_NUM_THREADS"] = "2"
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from lightgbm import LGBMClassifier
from scipy.interpolate import PchipInterpolator
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.signal import savgol_filter
from sklearn.inspection import PartialDependenceDisplay
from sklearn import metrics
from scipy.interpolate import make_interp_spline
import statsmodels.api as sm
# === 字体设置 ===
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# === 路径配置 ===
model_folder = r"G:\毕业设计\灾害易发性论文\临夏县及临夏市数据\模型终"
os.makedirs(model_folder, exist_ok=True)

# === 读取数据 ===
train_data = pd.read_csv(r"G:\毕业设计\灾害易发性论文\临夏县及临夏市数据\样本\训练集1.csv")
test_data_external = pd.read_csv(r"G:\毕业设计\灾害易发性论文\临夏县及临夏市数据\样本\测试集.csv")  # 假设这个是测试集

# 特征列和标签列
features = ['Lithology', 'Land_Use', 'Distance_to_Road', 'Annual_Precipitation',
            'Distance_to_River', 'Plan_Curvature', 'Profile_Curvature',
            'Aspect', 'Slope', 'NDVI', 'PGA', 'Soil_Type']
label = 'lx'
features_clean = [f.replace('_', ' ') for f in features]

# === 从原训练集中随机抽取100个样本作为一部分新测试集 ===
train_data_sampled = train_data.sample(n=100, random_state=1)  # 随机抽取100个
train_data_remaining = train_data.drop(train_data_sampled.index)  # 剩下的作为新的训练集

# === 合并成新的测试集 ===
new_test_data = pd.concat([test_data_external, train_data_sampled], ignore_index=True)

# 训练集
x_train_df = train_data_remaining[features]            # 原始 DataFrame
y_train = train_data_remaining[label]

imputer = SimpleImputer(strategy='mean')
X_imp = imputer.fit_transform(x_train_df)              # numpy array

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imp)

# 将 numpy 再转换回 DataFrame，保留列名
x_train = pd.DataFrame(X_scaled, columns=features)


# 测试集
x_test_df = new_test_data[features]
y_test = new_test_data[label]

X_imp_test = imputer.transform(x_test_df)
X_scaled_test = scaler.transform(X_imp_test)

# 同样转换回 DataFrame
x_test = pd.DataFrame(X_scaled_test, columns=features)



# === 交叉验证配置 ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === 基学习器定义 ===
base_learners = [
    ('RF', RandomForestClassifier(max_depth=6, min_samples_split=2,n_estimators=20,random_state=42)),
    ('LGBM', LGBMClassifier(n_estimators=100, force_col_wise=True, random_state=42, num_leaves=50, max_depth=4, min_child_samples=20, reg_alpha=0.0, reg_lambda=0.0, learning_rate=0.05)),
    ('SVM', SVC(probability=True, kernel='rbf', random_state=42)),
    ('MLP', MLPClassifier(hidden_layer_sizes=(128, 64 ,32), activation='relu', alpha=1e-3,
                          learning_rate='adaptive', learning_rate_init=0.0005, batch_size=64,  shuffle=True, tol=1e-5, solver='adam',early_stopping=True,validation_fraction=0.2, n_iter_no_change=30, max_iter=1000, random_state=42))
]

# === 绘制 ROC: 基模型 (5折 CV) ===
plt.figure(figsize=(8, 6))
for name, model in base_learners:
    y_proba = cross_val_predict(model, x_train, y_train, cv=cv, method='predict_proba')[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_train, y_proba)
    auc_score = metrics.roc_auc_score(y_train, y_proba)
    plt.plot(fpr, tpr, label=f"{name.replace('_', ' ')} (AUC={auc_score:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate', fontname='Times New Roman', fontsize=16)
plt.ylabel('True Positive Rate', fontname='Times New Roman', fontsize=16)
plt.title('Base Learners ROC Curve', fontname='Times New Roman',  fontsize=16)
plt.legend(loc='lower right', fontsize=14, prop={'family': 'Times New Roman'})
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(model_folder, 'base_models_roc.png'))
plt.close()

# === 自定义 BlendingModel ===
class BlendingModel(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"  # ← 添加这一行
    def __init__(self, base_models=None, meta_model=None, cv=5, weights=None):
        self.base_models = base_models
        self.meta_model = meta_model
        self.cv = cv
        self.weights = weights

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_models = len(self.base_models)
        n_classes = len(self.classes_)
        self.fitted_base_models_ = []
        meta_features = np.zeros((X.shape[0], n_models * n_classes))
        for i, (name, model) in enumerate(self.base_models):
            cloned = clone(model)
            proba = cross_val_predict(cloned, X, y, cv=self.cv, method='predict_proba')
            weight = self.weights[i] if self.weights else 1
            meta_features[:, i*n_classes:(i+1)*n_classes] = proba * weight
            cloned.fit(X, y)
            self.fitted_base_models_.append((name, cloned))
        self.meta_model_ = clone(self.meta_model).fit(meta_features, y)
        return self

    def predict_proba(self, X):
        meta_feats = np.hstack([
            model.predict_proba(X) * (self.weights[i] if self.weights else 1)
            for i, (_, model) in enumerate(self.fitted_base_models_)
        ])
        return self.meta_model_.predict_proba(meta_feats)

    def predict(self, X):
        meta_feats = np.hstack([
            model.predict_proba(X) * (self.weights[i] if self.weights else 1)
            for i, (_, model) in enumerate(self.fitted_base_models_)
        ])
        return self.meta_model_.predict(meta_feats)

# === 构建并训练全集成模型 ===
models = {}
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=cv
)
stacking_clf.fit(x_train, y_train)
models['Stacking'] = stacking_clf

simple_voting = VotingClassifier(estimators=base_learners, voting='soft')
simple_voting.fit(x_train, y_train)
models['Simple Averaging'] = simple_voting

# === 在构建模型前，搜索 WA 最佳权重（以 CV AUC 为准） ===
best_auc_wa = 0.0
best_w_wa = None
oof_probas = np.zeros((x_train.shape[0], len(base_learners)))
for i, (_, model) in enumerate(base_learners):
    oof_probas[:, i] = cross_val_predict(model, x_train, y_train, cv=cv,
                                         method='predict_proba')[:, 1]
for w1 in range(1, 4):
    for w2 in range(1, 4):
        for w3 in range(1, 4):
            for w4 in range(1, 4):
                w = np.array([w1, w2, w3, w4])
                proba = (oof_probas * w).sum(axis=1) / w.sum()
                auc_ = metrics.roc_auc_score(y_train, proba)
                if auc_ > best_auc_wa:
                    best_auc_wa = auc_
                    best_w_wa = [w1, w2, w3, w4]
print(f"最佳 WA 权重: {best_w_wa}, CV AUC: {best_auc_wa:.4f}")
# 定义专门用于 Blending 的交叉验证
blend_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
best_auc_bl = 0.0
best_w_bl = None
for w1 in range(1, 4):
    for w2 in range(1, 4):
        for w3 in range(1, 4):
            for w4 in range(1, 4):
                w = [w1, w2, w3, w4]
                blender_tmp = BlendingModel(
                    base_models=base_learners,
                    meta_model=LogisticRegression(max_iter=1000, solver='liblinear'),
                    cv=cv,
                    weights=w
                )
                probs = cross_val_predict(blender_tmp, x_train, y_train,
                                          cv=cv, method='predict_proba')[:, 1]
                auc_ = metrics.roc_auc_score(y_train, probs)
                if auc_ > best_auc_bl:
                    best_auc_bl = auc_
                    best_w_bl = w.copy()
print(f"最佳 Blending 权重: {best_w_bl}, CV AUC: {best_auc_bl:.4f}")

# === 用搜索到的最佳权重构建后续模型 ===
# Weighted Averaging
weighted_voting = VotingClassifier(
    estimators=base_learners,
    voting='soft',
    weights=best_w_wa
)
weighted_voting.fit(x_train, y_train)
models['Weighted Averaging'] = weighted_voting
# Blending
blending_model = BlendingModel(
    base_models=base_learners,
    meta_model=LogisticRegression(max_iter=1000, solver='liblinear'),
    cv=blend_cv,
    weights=best_w_bl
)
blending_model.fit(x_train, y_train)
models['Blending'] = blending_model
# === 保存所有模型 ===
fitted_base_learners = {}
for name, model in base_learners:
    fitted = clone(model).fit(x_train, y_train)
    fitted_base_learners[name] = fitted
    joblib.dump((fitted, imputer, scaler), os.path.join(model_folder, f"{name.lower()}_model.pkl"))
for name, model in models.items():
    key = name.replace(' ', '_').lower()
    joblib.dump((model, imputer, scaler), os.path.join(model_folder, f"{key}_model.pkl"))

# === 绘制 ROC: 集成模型 (5折 CV) ===
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_proba = cross_val_predict(model, x_train, y_train, cv=cv, method='predict_proba')[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_train, y_proba)
    auc_score = metrics.roc_auc_score(y_train, y_proba)
    plt.plot(fpr, tpr, label=f"{name.replace('_', ' ')} (AUC={auc_score:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate', fontname='Times New Roman', fontsize=16)
plt.ylabel('True Positive Rate', fontname='Times New Roman', fontsize=16)
plt.title('Ensemble Models ROC Curve', fontname='Times New Roman', fontsize=16)
plt.legend(loc='lower right', fontsize=14, prop={'family': 'Times New Roman'})
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(model_folder, 'ensemble_roc.png'))
plt.close()




# === 基模型 (测试集) 的 ROC：PCHIP 保单调平滑 ===
plt.figure(figsize=(8, 6))
for name, model in fitted_base_learners.items():
    y_proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    auc_score = metrics.roc_auc_score(y_test, y_proba)

    # 确保 fpr 单调且无重复
    fpr_u, idx = np.unique(fpr, return_index=True)
    tpr_u = tpr[idx]

    # PCHIP 插值（保单调）
    xnew = np.linspace(0, 1, 500)
    pchip = PchipInterpolator(fpr_u, tpr_u)
    tpr_smooth = pchip(xnew)

    plt.plot(xnew, tpr_smooth,
             label=f"{name.replace('_', ' ')} (AUC={auc_score:.3f})")

plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate', fontname='Times New Roman', fontsize=16)
plt.ylabel('True Positive Rate', fontname='Times New Roman', fontsize=16)
plt.title('Base Learners ROC Curve (Test Set)',
          fontname='Times New Roman', fontsize=16)
plt.legend(loc='lower right', fontsize=14, prop={'family': 'Times New Roman'})
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(model_folder, 'base_models_roc_test_pchip.png'))
plt.close()


# === 集成模型 (测试集) 的 ROC：同样处理 ===
plt.figure(figsize=(8, 6))
for name, model in models.items():
    y_proba = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
    auc_score = metrics.roc_auc_score(y_test, y_proba)

    fpr_u, idx = np.unique(fpr, return_index=True)
    tpr_u = tpr[idx]

    xnew = np.linspace(0, 1, 500)
    pchip = PchipInterpolator(fpr_u, tpr_u)
    tpr_smooth = pchip(xnew)

    plt.plot(xnew, tpr_smooth,
             label=f"{name.replace('_', ' ')} (AUC={auc_score:.3f})")

plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate', fontname='Times New Roman', fontsize=16)
plt.ylabel('True Positive Rate', fontname='Times New Roman', fontsize=16)
plt.title('Ensemble Models ROC Curve (Test Set)',
          fontname='Times New Roman', fontsize=16)
plt.legend(loc='lower right', fontsize=14, prop={'family': 'Times New Roman'})
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(model_folder, 'ensemble_roc_test_pchip.png'))
plt.close()




# === SHAP 样本选择：KMeans 聚类 ===

# 提取 SHAP 样本函数（保持不变）
def select_shap_samples(X, n_total=100, n_clusters=10):
    import os
    os.environ["OMP_NUM_THREADS"] = "2"
    X_arr = X.values
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(X_arr)
    labels = km.labels_
    samples = []
    per = max(1, n_total // n_clusters)
    for cid in range(n_clusters):
        grp = X.loc[labels == cid]
        if len(grp):
            np.random.seed(42)
            idx = np.random.choice(len(grp), min(per, len(grp)), replace=False)
            samples.append(grp.iloc[idx].values)
    return np.vstack(samples)[:n_total]

shap_sample = select_shap_samples(x_train, n_total=100, n_clusters=10)
# 保留列名，保证后续 SHAP explainer 接受 DataFrame
shap_df = pd.DataFrame(shap_sample, columns=features)
# === SHAP 分析与可视化（新版统一接口） ===
shap_results = {}
shap_models = {**fitted_base_learners, **models}

for name, model in shap_models.items():
    try:
        # 1) 构造 Explainer
        if isinstance(model, (RandomForestClassifier, LGBMClassifier)):
            expl = shap.TreeExplainer(model)
            raw = expl.shap_values(shap_df)             # 直接传 DataFrame
            vals = raw[1] if isinstance(raw, list) else raw
        else:
            def pred_fn(x):
                # x 进来就是 numpy array 或 DataFrame
                df = pd.DataFrame(x, columns=features)
                return model.predict_proba(df)
            expl = shap.KernelExplainer(pred_fn, shap_df.sample(50, random_state=42))
            out = expl.shap_values(shap_df, nsamples=100)
            vals = out[1] if isinstance(out, list) and len(out)==2 else out

        # 2) 构造 Explanation（data 用 numpy array）
        shap_exp = shap.Explanation(
            values=vals,
            base_values=np.mean(vals, axis=0),
            data=shap_df.values,
            feature_names=features_clean
        )

        # 3) Beeswarm
        plt.figure(figsize=(8,6))
        shap.plots.beeswarm(shap_exp, show=False)
        plt.title(f"{name} SHAP Summary")
        plt.subplots_adjust(left=0.2, right=0.95)
        plt.savefig(os.path.join(model_folder, f"shap_summary_{name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 4) Bar
        plt.figure(figsize=(8,6))
        shap.plots.bar(shap_exp, show=False)
        plt.title(f"{name} SHAP Feature Importance")
        plt.subplots_adjust(left=0.2, right=0.95)
        plt.savefig(os.path.join(model_folder, f"shap_bar_{name}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # 5) Top‑4 特征
        mean_abs = np.abs(shap_exp.values).mean(axis=0)
        top4 = np.argsort(mean_abs)[-4:][::-1]
        shap_results[name] = [features_clean[i] for i in top4]

        print(f"✅ SHAP 成功: {name}")

    except Exception as e:
        print(f"❌ SHAP 失败: {name} -> {e}")

# Top‑4 特征热力图
if shap_results:
    heat_df = pd.DataFrame([
        {'Model': m, 'Rank': f"Top {i+1}", 'Feature': feat}
        for m, feats in shap_results.items() for i, feat in enumerate(feats)
    ])
    heat_df['Model'] = pd.Categorical(heat_df['Model'], categories=list(shap_results.keys()), ordered=True)
    heat_df = heat_df.sort_values('Model')
    heat_matrix = heat_df.pivot(index='Model', columns='Rank', values='Feature')

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pd.DataFrame(1, index=heat_matrix.index, columns=heat_matrix.columns),
        annot=heat_matrix, fmt='', cmap='Greens', cbar=False,
        linewidths=.5, annot_kws={'fontsize':12}
    )
    plt.title('Top 4 SHAP Features per Model')
    plt.subplots_adjust(left=0.2, right=0.95)
    plt.savefig(os.path.join(model_folder, 'shap_top4_features_per_model.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print("✅ 热力图已保存")
else:
    print("⚠️ 无有效 SHAP 结果，跳过热力图")

# === 模型评估: 外部测试集指标 (包含 IOA) ===
def index_of_agreement(obs, pred):
    obs, pred = np.array(obs), np.array(pred)
    m = obs.mean()
    num = ((pred - obs)**2).sum()
    den = ((np.abs(pred - m) + np.abs(obs - m))**2).sum()
    return 1 - num/den if den != 0 else np.nan

results = []
for name, model in shap_models.items():
    # 直接在外部测试集上一次性预测
    y_pred = model.predict(x_test)
    results.append({
        'Model': name.replace('_', ' ').title(),
        'OA': round(metrics.accuracy_score(y_test, y_pred), 3),
        'Precision': round(metrics.precision_score(y_test, y_pred), 3),
        'Recall': round(metrics.recall_score(y_test, y_pred), 3),
        'F1': round(metrics.f1_score(y_test, y_pred), 3),
        'MCC': round(metrics.matthews_corrcoef(y_test, y_pred), 3),
        'IOA': round(index_of_agreement(y_test, y_pred), 3)
    })

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(os.path.join(model_folder, 'external_test_metrics.csv'), index=False)
print(metrics_df)
print("✅ 全部模型在外部测试集上完成评估并输出结果。")
# === 新增：计算 Brier 分数和 Ensemble 熵 ===
from sklearn.metrics import brier_score_loss
# Brier & 熵 结果列表
brier_results = []
entropy_results = []
for name, model in shap_models.items():
    y_proba_test = model.predict_proba(x_test)[:, 1]
    brier = brier_score_loss(y_test, y_proba_test)
    brier_results.append({'Model': name, 'BrierScore': brier})
    p = np.clip(y_proba_test, 1e-12, 1 - 1e-12)
    ent = -np.mean(p * np.log(p) + (1 - p) * np.log(1 - p))
    entropy_results.append({'Model': name, 'Entropy': ent})
# Brier 分数对比图
plt.figure(figsize=(8,4))
plt.bar([r['Model'].replace('_', ' ') for r in brier_results], [r['BrierScore'] for r in brier_results])
plt.xticks(rotation=45, ha='right')
plt.title('Brier Score Comparison', fontname='Times New Roman', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(model_folder, 'brier_score_comparison.png'))
plt.close()
# 熵对比图
plt.figure(figsize=(8,4))
plt.bar([r['Model'].replace('_', ' ') for r in entropy_results], [r['Entropy'] for r in entropy_results])
plt.xticks(rotation=45, ha='right')
plt.title('Ensemble Entropy Comparison', fontname='Times New Roman', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(model_folder, 'entropy_comparison.png'))
plt.close()



from sklearn.inspection import PartialDependenceDisplay
from sklearn.inspection import partial_dependence

# ———— 二、样式字典 ————
styles = {
    "RF":               {"color": "C0", "linestyle": "-"},
    "LGBM":             {"color": "C1", "linestyle": "--"},
    "SVM":              {"color": "C2", "linestyle": ":"},
    "MLP":              {"color": "C3", "linestyle": "-."},
    "Simple Averaging": {"color": "C4", "linestyle": "-"},
    "Weighted Averaging":{"color":"C5","linestyle":"--"},
    "Stacking":         {"color": "C6", "linestyle": ":"},
    "Blending":         {"color": "C7", "linestyle": "-."},
}

# ———— 三、特征名称准备 ————
features_clean   = [f.replace('_',' ') for f in features]
features_to_plot = features_clean[:12]

os.makedirs(model_folder, exist_ok=True)

# ———— 四、循环绘图 ————
for feat in features_to_plot:
    idx = features_clean.index(feat)
    # 1) 公共 grid
    pdp0 = partial_dependence(
        list(shap_models.values())[0],
        x_train,
        [idx],
        kind="average",
        grid_resolution=30
    )
    grid = pdp0["values"][0]
    xmin, xmax = grid.min(), grid.max()

    fig, ax = plt.subplots(figsize=(8,5))

    # 2) ICE 10–90% 区间带
    for name, mdl in shap_models.items():
        ice = partial_dependence(
            mdl, x_train, [idx],
            kind="individual", grid_resolution=30
        )["individual"][0]
        lower = np.percentile(ice, 10, axis=0)
        upper = np.percentile(ice, 90, axis=0)
        ax.fill_between(
            grid, lower, upper,
            color=styles[name]["color"], alpha=0.1
        )

    # 3) PDP 平均曲线
    for name, mdl in shap_models.items():
        avg = partial_dependence(
            mdl, x_train, [idx],
            kind="average", grid_resolution=30
        )["average"][0]
        ax.plot(
            grid, avg,
            label=name,
            color=styles[name]["color"],
            linestyle=styles[name]["linestyle"]
        )

    # 4) Rug plot
    for x in x_train.iloc[:, idx]:
        ax.axvline(x, ymin=0, ymax=0.02, color='k', alpha=0.05)

    # 5) 统一坐标轴
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1)

    # 6) 美化与保存
    ax.set_title(f"{feat} Comparison (PDP&ICE)", fontsize=14)
    ax.set_xlabel(feat, fontsize=12)
    ax.set_ylabel("Predicted Probability", fontsize=12)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()

    outfn = os.path.join(model_folder, f"PDP_ICE_comparison_{feat}.png")
    fig.savefig(outfn, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Saved: {outfn}")
