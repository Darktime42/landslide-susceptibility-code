import joblib
import pandas as pd
import os
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone
import numpy as np

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
# 定义模型路径
model_folder = r"G:\毕业设计\灾害易发性论文\临夏县及临夏市数据\模型一"
model_path = os.path.join(model_folder, "weighted_averaging_model.pkl")


# 加载模型
if not os.path.exists(model_path):
    raise FileNotFoundError(f"未找到模型文件 {model_path}，请先运行 train_model.py 进行训练！")

stacking_clf, imputer, scaler = joblib.load(model_path)

# 读取预测数据1'
test_data = pd.read_csv(r"G:\毕业设计\灾害易发性论文\临夏县及临夏市数据\样本\tifftocsv1.csv")

# 选择特征
features = ['Lithology', 'Land_Use', 'Distance_to_Road', 'Annual_Precipitation', 'Distance_to_River',
            'Plan_Curvature', 'Profile_Curvature', 'Aspect', 'Slope', 'NDVI', 'PGA', 'Soil_Type']
x_pred = test_data[features]

# 处理缺失值和标准化
# 处理缺失值并保持列名
x_pred_imputed = pd.DataFrame(
    imputer.transform(x_pred),
    columns=features
)

# 标准化（避免警告）：先转成 numpy，再还原为 DataFrame
x_pred_scaled = scaler.transform(x_pred_imputed.values)
x_pred = pd.DataFrame(x_pred_scaled, columns=features)

# 批量预测设置
batch_size = 10000  # 每次处理1000条数据，根据内存情况调整
n_samples = x_pred.shape[0]
pred_proba_list = []
pred_labels_list = []

# 按批次进行预测
for i in range(0, n_samples, batch_size):
    batch = x_pred[i:i + batch_size]

    # 进行批量预测
    batch_pred_proba = stacking_clf.predict_proba(batch)[:, 1]
    batch_pred_labels = stacking_clf.predict(batch)

    # 存储每批的结果
    pred_proba_list.extend(batch_pred_proba)
    pred_labels_list.extend(batch_pred_labels)

# 将所有预测结果合并
output_df_pred = pd.DataFrame({
    'Probability': pred_proba_list # 如果你需要保存标签
})

# 保存预测结果
output_folder = r"G:\毕业设计\灾害易发性论文\临夏县及临夏市数据\预测结果"
os.makedirs(output_folder, exist_ok=True)
output_df_pred.to_csv(os.path.join(output_folder, 'weighted_averaging_model_prediction_results.csv'), index=False)

print(f"预测结果已保存至 {os.path.join(output_folder, 'weighted_averaging_prediction_results.csv')}")
