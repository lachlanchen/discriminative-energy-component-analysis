# Run 2: additive observable contrast

Run 2 asks a broader question than classification:

> Which bounded physical observable changes most between two state streams,
> and can that observable be updated and interpreted online?

The reusable implementation is in [`../aoc/`](../aoc/). This run owns the
algebraic audits, streaming tests, Ising/order-mode experiment,
mass-spring-damage experiment, and optical/quantum simulation.

Run all run 2 tests:

```bash
.venv/bin/python -m pytest experiments/run2/tests -q
```

Run-specific scripts write only under `experiments/run2/results/`.

```bash
.venv/bin/python experiments/run2/scripts/run_algebraic_validation.py
.venv/bin/python experiments/run2/scripts/run_ising_order.py
.venv/bin/python experiments/run2/scripts/run_structural_monitoring.py
.venv/bin/python experiments/run2/scripts/run_optical_quantum.py
```

The compiled working paper is
[`../../publication/run2/main.pdf`](../../publication/run2/main.pdf).

The anytime-valid statement for the betting e-process assumes a known
independent reference state. Plug-in estimated references and overlapping
physical windows are reported as empirical diagnostics only.
