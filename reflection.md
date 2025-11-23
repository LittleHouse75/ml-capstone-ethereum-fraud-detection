# Reflection: From Benchmark Dataset to Real-World Scam Detection

## Accomplishments

This project has been very iterative. A lot of things broke along the way—timestamp issues, leakage scares, time splits that tanked performance—and I restarted more times than I’d like to admit. But I’m proud that I ended up with a clear story instead of just a pile of experiments.

I started with a single academic Ethereum dataset and a vague idea of “let’s model fraud.” The project eventually turned into: build address-level features, compare a **random address split** to a **time-based past→future split**, and then see whether the final model can say anything about a separate DFPI scam-wallet dataset. That arc only appeared after the time split broke my original results and forced me to rethink what problem I was actually solving.

I’m also happy that the model I pickled isn’t just a benchmark toy. The tuned XGBoost model trained on the benchmark data shows real predictive power on the DFPI scam-wallet set, which suggests it learned patterns that carry over into the real world, not just quirks of the academic data.

On the engineering side, I’m glad I took the time to clean up the code. The first notebook was huge and messy. By the end, I’d moved feature engineering, utilities, tuning, and evaluation into shared Python modules and split the work across several smaller notebooks (overview, EDA, random split, time split, external evaluation). That made the notebooks feel more like guided stories instead of giant code dumps, and it gave me a reusable pipeline for future datasets.

Underneath all of this, the main win for me is that I didn’t bail when the “easy” story fell apart. When the time split wrecked the model’s performance, I went back, fixed the logic, and accepted that the real story here was about drift and evaluation design. That felt like a real modeling lesson, not just a technical one.

## Opportunity for Growth

The biggest gap is the real-world dataset. Right now, the external evaluation focuses on DFPI-listed scam wallets plus some background traffic. It’s a solid proof of concept, but not yet the larger, more balanced, “production-like” dataset I’d want to really stress-test the model. With more time, I’d pull a larger, more representative set of **non-scam** addresses so I can see how the model behaves across everyday Ethereum activity, not just known bad actors.

One explicit stretch goal from my original pitch that I didn’t reach was experimenting with SGAN / synthetic data (an ATD-SGAN-style semi-supervised approach). I traded that away in favor of really understanding the time-based splits and the DFPI results. I still think synthetic data could be interesting here, but it would need more time and focus than I had in this cycle.

The pipeline, while much cleaner than where I started, is also more rigid than I’d like. It basically assumes XGBoost will be the main model. In a future iteration, I’d like a more flexible setup where I can point to a training dataset and a test dataset, have a small family of models trained and compared automatically, and then pick the winner based on the same metrics, instead of relying on my early assumption that XGBoost is “the one.”

Finally, there’s an opportunity to tell the “false starts” story more clearly. The timestamp parsing problems, the near-leakage, the first disastrous time split—those are a big part of why I now care so much about time-aware splits and careful preprocessing. They’re only hinted at in the current write-up. Before final submission, I’d like to surface those a bit more and connect them to what I’d change next time.

## Feedback Request

The main thing I’d like feedback on is structure.

I ended up with multiple notebooks (overview, EDA, random split, time split, DFPI evaluation) and pushed most of the heavy lifting into `.py` files. From my side, that made the work easier to manage and reuse, but I’m curious how it lands for other people.

As a reader, does this setup make the project easier to follow, or does it feel like too much is “hidden” in the modules? Do the notebooks feel like clear narratives that call well-named helpers, or do you find yourself wishing more of the core logic—especially the feature engineering and split logic—were visible directly in the notebooks?

Any specific suggestions on where you’d like to see more code inline, or where the random vs time split and DFPI evaluation could be explained more clearly, would be really helpful as I polish the final version.