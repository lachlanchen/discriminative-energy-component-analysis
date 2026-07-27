# From ECA to accessible-observable contrast

## Executive conclusion

The most defensible generalization of Eigen-Component Analysis is not another
variance criterion. It is an **accessible-observable discrimination problem**:

> Encode each observation as a positive state, specify which measurements are
> physically or computationally accessible, and choose the accessible
> measurement whose expectation separates the states as much as possible.

This formulation answers the original motivation—use maximum difference
between classes rather than maximum variance—without tying the idea to a
particular classifier. It also separates three layers that were previously
mixed:

1. **representation:** what information is retained in the positive state
   \(R(x)\);
2. **physics or invariance:** which observables are allowed;
3. **decision:** which allowed observable best distinguishes states.

For two classes and unrestricted bounded effects, the answer is the positive
spectral projector of the density difference. For multiple classes, the exact
extension is a positive-operator-valued measurement (POVM) optimization. For a
symmetry or measurement restriction, the density difference is first
projected into the accessible observable algebra. Streaming and federated
updates are exact because the sufficient statistics are additive.

The individual mathematical ingredients are established results from quantum
state discrimination, operator theory, invariant testing, kernel embeddings,
and sequential analysis. The research opportunity is the disciplined
combination: mergeable learned witnesses, capacity-constrained effects,
symmetry-sector diagnostics, and physical validation in regimes where mean or
variance alone is blind.

## 1. State representation rather than a covariance slogan

Let \(\phi(x)\in\mathbb C^d\) be a nonzero feature map and define

\[
R(x)=\frac{\phi(x)\phi(x)^\dagger}
           {\|\phi(x)\|_2^2},\qquad
R(x)\succeq0,\qquad \operatorname{Tr}R(x)=1.
\]

For class or regime \(c\), retain only

\[
S_c=\sum_i w_i R(x_i),\qquad
m_c=\sum_iw_i,\qquad
\rho_c=S_c/m_c.
\]

The pair \((S_c,m_c)\) is a mergeable sufficient statistic for every later
linear observable query \(\operatorname{Tr}(O\rho_c)\). It supports:

- one-at-a-time addition: \(S\leftarrow S+wR(x)\);
- exact distributed merge: \((S,m)=(S_A+S_B,m_A+m_B)\);
- deletion and sliding windows when the removed state is retained;
- exponential forgetting: \((S,m)\leftarrow(\gamma S,\gamma m)\);
- delayed choice of the final observable.

This is stronger and more precise than saying that ECA “uses covariance.”
The map \(R\) determines the retained information. With \(\phi(x)=x\), sign
and global phase disappear, which is ideal for phase-blind power changes but
fatal when phase carries the label.

Higher-order information can be retained without losing additivity:

\[
R_p(x)=
\frac{\phi(x)^{\otimes p}\phi(x)^{\otimes p\dagger}}
     {\|\phi(x)\|^{2p}}.
\]

Its Hilbert--Schmidt inner product is the degree-\(2p\) projective kernel
\(|\langle\hat\phi(x),\hat\phi(z)\rangle|^{2p}\). The price is dimension
\(d^p\), so symmetric tensor compression or structured tensor networks are
required beyond small \(p\).

## 2. Binary maximum difference has an exact operational answer

For \(\Delta=\rho_1-\rho_0\), choose an effect \(0\preceq E\preceq I\). Its
mean separation is

\[
\delta_E=\operatorname{Tr}(E\Delta).
\]

By the Jordan decomposition
\(\Delta=\Delta_+-\Delta_-\), with orthogonal positive parts,

\[
\max_{0\preceq E\preceq I}\operatorname{Tr}(E\Delta)
=\operatorname{Tr}\Delta_+,
\qquad
E^\star=\mathbf 1_{\Delta>0}.
\]

When \(\operatorname{Tr}\Delta=0\),

\[
\operatorname{Tr}\Delta_+
=\frac12\|\Delta\|_1.
\]

Thus “maximum class difference” is the trace distance between the two
empirical states. It is an integral probability metric over bounded quadratic
observables, not a variance maximum:

\[
\|\rho_1-\rho_0\|_1
=\sup_{\|O\|_\infty\le1}
  \left|\mathbb E_1\,\operatorname{Tr}[O R(x)]
       -\mathbb E_0\,\operatorname{Tr}[O R(x)]\right|.
\]

With equal priors, the corresponding one-shot discrimination success is

\[
P_{\rm succ}^{\star}
=\frac12+\frac14\|\rho_1-\rho_0\|_1.
\]

This is the Helstrom theorem. It should be credited as such; presenting the
spectral solution as newly invented would be wrong. What it supplies here is
the missing operational interpretation of the original ECA objective.

### Capacity-constrained measurements

If the instrument may retain at most \(r\) modes,

\[
\max_{\substack{0\preceq E\preceq I\\\operatorname{rank}E\le r}}
\operatorname{Tr}(E\Delta)
=\sum_{j=1}^{r}\max(\lambda_j(\Delta),0).
\]

The optimal effect selects the \(r\) largest positive eigenmodes. This Ky Fan
restriction is not merely regularization: it models a real measurement or
storage budget. In the structural and robot simulations, rank one removes
irrelevant positive modes and recovers the planted physical damage/contact
mode.

## 3. The correct multiclass object is a POVM

For \(K>2\), a collection of independent one-versus-rest projectors need not
form a valid joint measurement. The coherent multiclass problem is

\[
\begin{aligned}
\max_{\{E_c\}_{c=1}^K}\quad&
\sum_{c=1}^K\pi_c\operatorname{Tr}(E_c\rho_c),\\
\text{subject to}\quad&
E_c\succeq0,\qquad \sum_cE_c=I.
\end{aligned}
\]

This is the known minimum-error quantum-state-discrimination SDP. Its dual is

\[
\min_Y\operatorname{Tr}Y
\quad\text{subject to}\quad
Y\succeq\pi_c\rho_c\ \text{for all }c.
\]

It gives:

- an operational multiclass score, \(P_{\rm succ}^{\star}\);
- a no-information baseline, \(\max_c\pi_c\);
- class-specific positive effects that sum to one;
- a certificate from primal--dual agreement.

Binary equal-prior discrimination reduces to the positive/negative spectral
measurement above. General multiclass discrimination usually has no single
eigendecomposition solution, which is an important boundary for claims about
ECA.

## 4. The deeper theorem: project onto the accessible algebra

Suppose physical constraints permit only effects in a finite-dimensional
unital \(*\)-subalgebra \(\mathcal A\subseteq M_d\). Let
\(\mathcal E_{\mathcal A}\) be the trace-preserving conditional expectation
onto \(\mathcal A\). For every \(E\in\mathcal A\),

\[
\operatorname{Tr}(E\Delta)
=\operatorname{Tr}\!\left[E\mathcal E_{\mathcal A}(\Delta)\right].
\]

Therefore

\[
\boxed{
\sup_{\substack{E\in\mathcal A\\0\preceq E\preceq I}}
\operatorname{Tr}(E\Delta)
=\operatorname{Tr}
  \left[\mathcal E_{\mathcal A}(\Delta)_+\right]
}
\]

and the optimal accessible effect is the support projector of the positive
part inside \(\mathcal A\).

This is the most general finite-dimensional statement developed here.
Symmetry, locality, sensor bandwidth, block-diagonal instruments, and
coarse-grained readout are all choices of \(\mathcal A\). The accessible trace
distance cannot exceed the unrestricted distance. That is an operational form
of information loss under coarse graining.

### Symmetry as a special case

For a compact group representation \(U_g\), the accessible algebra is the
commutant \(\{U_g\}'\), and the conditional expectation is group twirling:

\[
\mathcal T_G(X)=\int_GU_gXU_g^\dagger\,dg.
\]

Hence

\[
\sup_{\substack{0\preceq E\preceq I\\[E,U_g]=0}}
\operatorname{Tr}(E\Delta)
=\operatorname{Tr}[\mathcal T_G(\Delta)_+].
\]

Under
\(\mathcal H=\bigoplus_\lambda V_\lambda\otimes M_\lambda\),
Schur's lemma yields blocks

\[
\mathcal T_G(\Delta)
=\bigoplus_\lambda
  \frac{I_{V_\lambda}}{\dim V_\lambda}\otimes\Delta_\lambda.
\]

The trace norm and positive gap decompose across symmetry sectors. This turns
a detector into a diagnostic: it reports not only that a state changed, but
which charge, parity, frequency, angular-momentum, or representation sector
carried the change.

## 5. Relation to neighboring methods

### PCA and SVD

PCA maximizes total second moment,
\(\max_{\|v\|=1}v^\dagger\rho v\). It does not use labels and can select a
high-variance nuisance direction. Observable contrast diagonalizes a
difference of states. The two coincide only in special cases, such as a
zero/reference background proportional to identity.

### CSP and generalized eigenvectors

Common spatial patterns optimize variance ratios such as
\(v^\dagger C_1v/v^\dagger C_0v\), leading to a generalized eigenproblem. AOC
optimizes a bounded difference after trace normalization. CSP is appropriate
for relative power under a reference metric; AOC is appropriate when absolute
expectation difference and a bounded measurement have direct meaning.

### Kernel MMD

For the projective kernel
\(k(x,z)=|\langle\hat\phi(x),\hat\phi(z)\rangle|^2\),

\[
\operatorname{MMD}_k^2(P,Q)
=\|\rho_P-\rho_Q\|_F^2.
\]

MMD uses the Hilbert--Schmidt norm of the same operator difference. AOC uses
its trace norm and additionally returns the optimal bounded witness. Neither
norm universally dominates for statistical power; the spectrum and
regularization determine the better finite-sample test.

### Energy-based models and Boltzmann machines

An RBM learns an entire generative energy landscape with latent stochastic
units, usually by approximate likelihood optimization. AOC learns one
discriminative bounded observable analytically from additive states. Their
goals and computational burdens differ. A possible bridge is to use the
learned witness as an initialization or diagnostic order parameter for an
energy-based model, not to claim they are the same algorithm.

## 6. Streaming inference without invalid adaptivity

A witness fitted on the same sample on which it is evaluated is optimistically
biased. A valid sequential construction must make the witness predictable:
\(E_t\) is a function only of observations before time \(t\).

For a known reference state \(\rho_0\), define a bounded score
\[
z_t=\operatorname{Tr}[E_t(R_t-\rho_0)]\in[-1,1].
\]
Under an independent stationary null,
\(\mathbb E[z_t\mid\mathcal F_{t-1}]=0\). For fixed
\(\lambda\in(0,1)\),
\[
M_t(\lambda)=\prod_{i=1}^t(1+\lambda z_i)
\]
is a nonnegative martingale. Mixtures over betting fractions and candidate
change times remain e-processes, so Ville-type thresholds provide anytime
control under the stated null.

Two limitations matter:

1. replacing the known \(\rho_0\) by a plug-in estimate generally destroys the
   exact guarantee unless estimation uncertainty is incorporated;
2. adaptive forgetting and overlapping windows introduce dependence that must
   be modeled or calibrated.

The stored null experiments are empirical audits, not substitutes for these
assumptions.

## 7. Where the method can be decisive

The method is not universally superior. It is decisive when four conditions
hold:

1. the class/regime difference lives in the encoded operator moment;
2. the raw mean is zero or dominated by nuisance variation;
3. the relevant effect is low rank or symmetry restricted;
4. samples arrive incrementally or across sites, making additive summaries
   valuable.

The repository contains predeclared matched cases:

- **Ising order:** exact global \(Z_2\) pairing forces every linear mean
  classifier to chance, while the learned quadratic observable recovers the
  magnetization mode and reaches approximately \(99.98\%\) accuracy.
- **Polarization:** horizontal/vertical readout is exactly blind to diagonal
  polarization states, while the learned Helstrom analyzer raises theoretical
  success from \(50\%\) to \(95\%\), reproduced with finite-shot Aer
  simulation.
- **Structural damage:** zero-mean thermal excitation hides the damage from a
  window-mean classifier; a rank-one witness reaches about \(0.977\) ROC AUC
  and aligns with the analytical damage mode.
- **Cyclic translation:** with one sign-paired training example per class,
  raw-space methods are at chance while the invariant AOC and the correctly
  specified Fourier-power baseline are essentially perfect.
- **Robot contact:** in a controlled six-axis wrench model, the learned
  rank-one effect overlaps the planted contact screw by about \(0.999\);
  deployment claims require real robot data.

These examples prove the value of matching the observable algebra to the
problem. They do not prove generic superiority over RBF kernels, engineered
Fourier features, likelihood-ratio tests, or domain-specific estimators.

## 8. Physics, chemistry, optics, vision, and robotics

### Statistical and condensed-matter physics

The leading witness can be a data-driven order parameter. Sector-resolved
trace distance gives a finite-system response function across a parameter
scan. In the ten-qubit transverse-field Ising calculation, the reduced-state
response peaks near the known \(g/J=1\) transition and separates even/odd
parity contributions. This diagnoses a known model; it is not a new solution
of it.

### Optics

A normalized \(2\times2\) coherency matrix is mathematically a qubit density
matrix. The optimal effect maps directly to a polarization analyzer: its
Bloch vector gives wave-plate and polarizer settings. For spatial coherence,
larger cross-spectral density matrices support the same eigenmode analysis.
Practical work must include shot noise, detector calibration, nonunitary loss,
and analyzer constraints.

### Chemistry

The one-particle difference density
\(\Delta\gamma=\gamma_{\rm product}-\gamma_{\rm reactant}\) has positive and
negative natural difference orbitals. These are attachment and detachment
modes, exactly the Jordan decomposition used by AOC. The Hückel ring example
validates conservation and interpretation, but quantitative chemistry needs
correlated electronic-structure calculations, a nonorthogonal atomic-orbital
metric when applicable, and comparison with established natural transition
orbital/difference-density tools.

For atomistic environments, rotationally invariant algebras lead toward
spherical-harmonic power spectra and bispectra. SOAP and ACE already provide
strong, systematic constructions. AOC's possible role is supervised selection
and streaming comparison of those invariant density features, not replacement
without benchmarks.

### Vision and signals

Cyclic translation twirling is diagonal in the DFT basis and can be updated
from power spectra in \(O(d\log d)\) per signal. It is attractive for vibration
monitoring, periodic textures, radar/sonar range profiles, and optical speckle
when translation or phase is nuisance. Power spectra cannot distinguish
homometric signals or phase-coded structure; bispectral or equivariant
representations are then necessary.

### Robotics

Candidate states include force/torque residual covariances, tactile taxel
correlations, joint-space innovation covariances, and visual feature-density
operators. A low-rank effect can localize a contact screw or a compliance
change while retaining constant-size online summaries. Required practical
tests include changing payload, pose-dependent coordinate transforms,
heteroscedastic sensor noise, unseen contact geometry, latency, and public or
hardware datasets.

## 9. What the string-theory connection can and cannot mean

There is a legitimate mathematical bridge, but no present claim of solving a
string-theory problem.

### Legitimate bridge

Quantum field theory, lattice gauge theory, tensor networks, and holography use
states restricted to subregions or observable algebras. Symmetry and gauge
constraints decompose reduced states into charge sectors. Given two computed
states \(\rho_A(\theta)\) and \(\rho_A(\theta+\delta)\), accessible-observable
contrast asks:

\[
\text{Which bounded, gauge-invariant operator in region \(A\) best
distinguishes the two backgrounds or couplings?}
\]

Sector resolution can identify whether distinguishability is carried by
charge, parity, representation, or edge-mode blocks. Tensor-network
representations may permit matrix-free estimates of leading positive modes,
and classical-shadow data may estimate selected observables experimentally.

### Relation to modular theory

For a reference reduced state \(\rho\), the modular Hamiltonian is
\(K=-\log\rho\). The entanglement first law relates an infinitesimal entropy
change to \(\delta\langle K\rangle\). AOC instead optimizes over bounded
effects and returns \(\operatorname{sign}(\delta\rho)\) or its positive
projector. These are different questions:

- \(K\) is a distinguished, generally unbounded state-dependent observable;
- AOC is the best bounded discriminator under an operational norm.

Comparing them could reveal when the modular response is also the most
operationally distinguishable response, but equality should not be assumed.

### Essential cautions

- Gauge theories do not always factorize cleanly into spatial tensor products;
  the observable algebra and treatment of edge modes must be specified.
- Holographic states are often available only indirectly or perturbatively.
- Trace-distance estimation is hard in exponentially large Hilbert spaces.
- A numerical sector peak is not evidence for a new string vacuum,
  duality, or phase unless tied to a concrete accepted model and independent
  checks.

A credible first project would use a small lattice gauge theory or a published
matrix-product-state dataset, predeclare two couplings/phases, compute
gauge-invariant reduced states, and test whether the learned sector witness
agrees with known Wilson-loop, flux, or order diagnostics.

## 10. New research program and falsifiable milestones

### A. Algebra-aware AOC

Implement conditional expectations for block, locality, and instrument
algebras. Prove finite-sample perturbation bounds for the accessible positive
projector. Falsification: if the accessible eigengap is small, learned witness
orientation should be declared unstable even when the scalar distance is
stable.

### B. Certified multiclass ECA

Use the POVM SDP with primal--dual certificates, then study low-rank,
commuting, and tensor-structured restrictions. Compare against one-versus-rest
ECA on calibration, storage, and decision risk. Falsification: no advantage
over prior guessing on identical states and exact reduction to Helstrom in the
binary case.

### C. Higher-order but additive states

Use symmetric tensor sketches to retain phase-sensitive third/fourth-order
information. Compare power-spectrum failures against bispectral recovery.
Falsification: construct homometric signals that the second-order state cannot
separate and verify that the reported distance is zero.

### D. Matrix-free many-body and chemistry solvers

Estimate leading eigenpairs of \(\Delta\) from tensor-network contractions or
electronic-structure density matrices without materializing \(d^2\) entries.
Validate against exact small systems before scaling.

### E. Real streaming systems

Evaluate prequentially on a public structural, optical, EEG, or robot contact
stream. Freeze the reference, update witnesses only from the past, report
average run length, delay, mode stability, compute, and calibration. A
simulation-only result is not enough for deployment.

## 11. Naming and claim discipline

“Eigen-Component Analysis” is historically useful but underspecifies the
general object. A defensible hierarchy is:

- **ECA:** the original supervised eigencomponent model;
- **DECA:** the tested discriminative operator formulation in run 1;
- **AOC:** additive observable contrast in run 2;
- **SAOC:** symmetry-/subalgebra-accessible observable contrast in run 3.

The strongest concise claim is:

> AOC turns additive positive-state summaries into the bounded observable that
> maximizes empirical regime contrast; SAOC does the same after projecting onto
> the physically accessible observable algebra.

That claim is exact, useful outside machine learning, and narrow enough to
survive serious review.

## Primary literature map

- C. W. Helstrom, *Quantum Detection and Estimation Theory* (Academic Press,
  1976): binary minimum-error discrimination and trace distance.
- A. S. Holevo, *Probabilistic and Statistical Aspects of Quantum Theory*
  (1982): quantum statistical decision theory.
- F. Hiai, M. Mosonyi, and M. Hayashi, “Quantum hypothesis testing with group
  symmetry,” arXiv:0904.0704: invariant measurement restrictions.
- M. Goldstein and E. Sela, “Symmetry-resolved entanglement in many-body
  systems,” *Physical Review Letters* 120, 200602 (2018).
- L. Parra and P. Sajda, “Blind Source Separation via Generalized Eigenvalue
  Decomposition,” *JMLR* 4, 1261–1269 (2003): covariance generalized
  eigenproblems and the CSP connection.
- M. Zecchin, O. Simeone, and A. Ramdas, “Universal Sequential Changepoint
  Detection of Quantum Observables via Classical Shadows,” arXiv:2602.11846
  (2026): current quantum-observable sequential detection; it prevents a broad
  novelty claim for generic online quantum change detection.
