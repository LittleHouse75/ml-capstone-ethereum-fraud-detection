![Banner](https://github.com/LittleHouse75/flatiron-resources/raw/main/NevitsBanner.png)

# Ethereum Scam Address Modeling

End-to-end **address-level scam risk modeling on Ethereum**: build behavioral features from transactions, train a supervised model, and test how well it holds up under **time drift** and **dataset shift** (including an external regulator list: **CA DFPI**).

## Quick links

- **Notebook Dashboard:** https://github.com/LittleHouse75/flatiron-ml-modeling-pipeline/blob/main/notebooks/00_Overview.ipynb  
- **Pitch:** https://github.com/LittleHouse75/flatiron-ml-modeling-pipeline/blob/main/flatiron-ml-modeling-pitch.md  
- **Reflection:** https://github.com/LittleHouse75/flatiron-ml-modeling-pipeline/blob/main/reflection.md  
- **Video Presentation:** https://youtu.be/31tma5EMXwE  

## Repo tour

- `notebooks/` — EDA, feature engineering, modeling, and evaluations (see dashboard link above)
- `src/` — feature engineering + reusable pipeline code (ex: `src/featureeng.py`)
- `data/` — expected inputs / generated artifacts (including DFPI-related files, if present)
