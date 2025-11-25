# Reflection: From Benchmark Dataset to Real-World Scam Detection

### Accomplishments

The project moved through several restarts—timestamp bugs, leakage scares, and a time split that demolished my early results. The upside is that those detours forced the work into an actual narrative instead of a jumble of experiments.

I began with a single academic Ethereum dataset and a loose idea: *model fraud.* Over time, that turned into a clearer progression:
build address-level features → test a **random address split** → test a **past→future time split** → evaluate the final model on a **real regulator dataset (DFPI)**.
That last piece only emerged because the time split exposed how fragile the original story was. Fixing that pushed me to rethink the problem in terms of drift and transferability.

The tuned XGBoost model I ended up with isn’t just a benchmark toy. Applied to the DFPI dataset—where scams are vanishingly rare—the model still shoves almost all DFPI-reported scams into the extreme right tail of the score distribution. That tells me it learned portable behavioral patterns rather than memorizing quirks of the academic data.

On the engineering side, I cleaned up a lot. The original notebook was enormous. By the end, feature engineering, utilities, tuning, and evaluation lived in shared modules, and the notebooks were split into a small sequence: overview, EDA, random split, time split, external evaluation. That shift made the whole project feel more like a set of explanations than a giant code dump, and it gave me a reusable structure for future datasets.

The thing I’m most satisfied with is not giving up when the simple version collapsed. The failed time split made me slow down, fix the underlying logic, and accept that the real story here was about drift and evaluation design—not just hitting a high AUC.

### Opportunity for Growth

The main gap is scope. The DFPI dataset is valuable, but it’s still narrow: almost all addresses in that slice are neighbors of known scams. It’s a good stress test but not yet a broad, production-like “everyday Ethereum” dataset. With more time, I’d build a larger pool of **non-scam** addresses so I could measure how often the model fires on normal traffic.

I also dropped one of my original ambitions: trying SGAN or other semi-supervised approaches. Once the time split blew up, I traded that exploration for deeper work on drift and external evaluation. I still want to revisit synthetic/representation-learning methods, but they’d need their own focused cycle.

The pipeline itself is cleaner now, but it’s rigid. It assumes XGBoost is the main model. Next iteration, I want a more flexible system that can run a family of models end-to-end, compare them, and pick a champion without that early bias.

Another thing worth surfacing more clearly is the set of mistakes: the timestamp parsing problems, the early leakage risk, the failed time split. Those missteps shaped the final design, and they’re underrepresented in the current write-up.

### Continual Improvement

If I were turning this into a longer-running system, I’d focus on five areas:
* **Richer, evolving label sources.**

⠀Aggregate scam labels from multiple places—Etherscan tags, public blacklists, regulator notices, phishing-report feeds—and keep track of when labels first appear. That becomes a growing master list instead of a fixed snapshot.
* **Broader background data.**

⠀Sample larger sets of *non-scam* addresses from general Ethereum activity, not just neighbors of known scams. That would let me measure alert rates and stability across regular traffic.
* **Automated retraining and monitoring.**

⠀Build a cloud pipeline that ingests new scam labels and transactions over time, refreshes features, retrains models as drift accumulates, and tracks alert-rate behavior on recent blockchain slices.
* **Model and methodology expansion.**

⠀Revisit semi-supervised approaches and explore graph-based models on the address–transaction network. Compare them against the tabular XGBoost baseline inside the same evaluation harness.
* **Multi-chain generalization.**

⠀Extend the feature engineering to other chains—first EVM-compatible, then non-EVM chains with modified features. Aim for a reusable “fraud modeling” toolkit rather than an Ethereum-only pipeline.

### Feedback Request

I split the work into several notebooks and pushed most logic into .py modules. From my side, that made the project easier to manage, but I’m curious how it reads.

Do the notebooks feel like clear explanations that call well-named helpers, or does the separation hide too much of the core logic?
Would a reader want to see more of the split logic and feature engineering inline, or is the modular structure clearer?

Any thoughts on where the random vs. time split and DFPI evaluation could be more explicit would help as I polish the final pass.