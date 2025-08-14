# 11 — Hybrid Ensembles

Goal: Combine neural nets, tree models, and linear models to leverage complementary strengths.

## Patterns
- Feature-level fusion: concatenate learned representations (e.g., CNN embeddings) with tabular features → boosted trees.
- Model-level stacking: train separate models per modality and stack with a meta-learner.

## Use case — Stacking CNN + XGBoost for vision+tabular
- Vision: CNN extracts image embeddings.
- Tabular: numeric/categorical features.
- Stacking: base models → meta logistic/linear model.

### Sketch (PyTorch + XGBoost)
```python
# 1) Extract CNN embeddings
import torch, torchvision as tv
from torch import nn

model = tv.models.resnet18(weights=tv.models.ResNet18_Weights.DEFAULT)
backbone = nn.Sequential(*(list(model.children())[:-1]))  # global avg pool output (512-d)
backbone.eval()

def extract_embed(img):
    with torch.no_grad():
        x = backbone(img.unsqueeze(0))
        return x.squeeze().cpu().numpy()

# 2) Build training matrix: [cnn_embed | tabular_features]
# X_img_embed: shape (n, 512); X_tab: shape (n, p)
# X_all = np.hstack([X_img_embed, X_tab])

# 3) Train XGBoost on concatenated features
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8)
xgb.fit(X_all, y)
```

### Stacking variation
```python
# Train separate models: CNN classifier, XGBoost on tabular.
# Get out-of-fold probabilities p_img, p_tab (K-fold).
# Meta = LogisticRegression on [p_img, p_tab] (and optionally key raw features).
```

## Tips
- Normalize embeddings and tabular features; reduce embedding dimensionality (PCA) if needed.
- Ensure synchronization between images and tabular rows; avoid leakage across folds.
- For NLP, replace CNN with Transformer embeddings (e.g., sentence-BERT) and repeat.
