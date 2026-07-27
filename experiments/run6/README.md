# Run 6: real-QEC S-PACE audit

Run 6 tests whether predictable maximum-difference correlation witnesses add
anything beyond detector firing rate, detector likelihood, covariance
monitoring and a same-feature online logistic witness on real QEC records.

The primary dataset is the Google Quantum AI 2022 distance-25 repetition-code
experiment. Its archive README identifies a high-energy event near shot
57,775. The metadata, theory, feature map, splits, hyperparameters and
advantage rule are now frozen before held-value access. Detector values and
decoder outcomes remain held until the implementation, freeze-manifest and
ratification commits are pushed and verified.

Evidence classes remain separate:

- exact design-based random-swap validation on real observed pairs;
- causal replay of the approximately located Google hardware event;
- constructed boundaries between PNNL/IBM hardware snapshot cohorts; and
- any later controlled simulation.

The complete preregistration is
[`references/run6_real_qec_preregistered_plan.md`](../../references/run6_real_qec_preregistered_plan.md).
The final theory is
[`references/run6_space_final_theory.md`](../../references/run6_space_final_theory.md).

Raw public data are downloaded to the ignored `experiments/data/run6/`
directory. Only download scripts, checksums, configurations, derived result
tables and manifests are committed.

The executable order is:

1. create and push the implementation, freeze manifest and ratification;
2. run the detector-only Google replay;
3. run 32 deterministic randomization shards and their exact merge;
4. run the independent Pittsburgh snapshot arm; and
5. join only Google outcome records 40,000--59,999 and evaluate the locked
   Boolean.

No current result supports an S-PACE advantage. The locked decision becomes
positive only if the Google event result and the independent auxiliary arm
both pass the predeclared same-budget gate. Otherwise the required conclusion
is “no demonstrated S-PACE algorithmic advantage.”
