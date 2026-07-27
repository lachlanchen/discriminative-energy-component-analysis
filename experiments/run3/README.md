# Run 3: symmetry-resolved observable contrast

Run 3 restricts the detector to observables compatible with a known physical
symmetry. Group twirling is performed before the Jordan/Ky Fan solution, and
the final distinguishability is decomposed into symmetry sectors.

Exact cases in this run include cyclic translations/Fourier power, finite
groups, a transverse-field Ising parity decomposition, and a controlled
Hückel difference-density example.

```bash
.venv/bin/python -m pytest experiments/run3/tests -q
```

All generated evidence belongs under `experiments/run3/results/`.

```bash
.venv/bin/python experiments/run3/scripts/run_translation_vision.py
.venv/bin/python experiments/run3/scripts/run_quantum_phase.py
.venv/bin/python experiments/run3/scripts/run_huckel_difference.py
.venv/bin/python experiments/run3/scripts/run_robot_contact.py
```

The compiled working paper is
[`../../publication/run3/main.pdf`](../../publication/run3/main.pdf).

The chemistry and robot experiments are controlled physical models. The
quantum-field/string-theory discussion is a concrete future reduced-state
calculation, not an empirical result in this run.
