# LVAD VA Model

This repository contains the open model weights for the **LVAD VA Model**, a pair of gradient boosted survival models predicting post-LVAD outcomes from baseline characteristics.

### Cohort
- Source dataset: INTERMACS Registry  
- Years: 2008–2019

### Models
Two models exist:

| model | file | description |
|---|---|---|
| early | `models/xgbearlyVA_survival_imb_ex.pkl` | trained using early-event window |
| late | `models/xgblateVA_survival_imb.pkl` | trained using late-event window |

### Quick Test

```bash
export PYTHONPATH=src
python - <<EOF
from my_model.model import feature_names, predict
import pandas as pd

r = pd.DataFrame([{c:0 for c in feature_names('early')}])
print("EARLY:", predict(r, which="early"))
EOF

