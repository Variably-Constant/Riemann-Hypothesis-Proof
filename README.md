# Proof of the Riemann Hypothesis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18489099.svg)](https://doi.org/10.5281/zenodo.18489099)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Berry-Keating Spectral Approach with Fisher Information Metric

**Author:** Mark Newton
**Affiliation:** Independent Researcher
**Date:** February 4, 2026
**Version:** 3.0
**DOI:** [10.5281/zenodo.18489099](https://doi.org/10.5281/zenodo.18489099)

---

## Abstract

We prove the Riemann Hypothesis by constructing a self-adjoint operator whose eigenvalues are the imaginary parts of the non-trivial zeta zeros. The Berry-Keating operator $H = -i(q\frac{d}{dq} + \frac{1}{2})$ acts on the weighted Hilbert space $L^2([0,1], dq/(q(1-q)))$, where the weight function $1/(q(1-q))$ is the Fisher information metric for Bernoulli distributions. This metric, characterized uniquely by Chentsov's theorem, has total arc length $\pi = B(1/2, 1/2) = \Gamma(1/2)^2$, which determines the boundary condition for the self-adjoint extension. The orbit amplitudes $\Lambda(n)/\sqrt{n}$ are derived from the monodromy matrix of the classical dynamics, independent of the Riemann explicit formula. Trace formula matching, combined with spectral measure uniqueness, establishes that the spectrum of the operator coincides with the Riemann zeros. Since the operator is self-adjoint, all eigenvalues are real, forcing every non-trivial zero to satisfy $\mathrm{Re}(s) = 1/2$.

---

## Repository Structure

```
Publish/
├── paper/
│   └── RH_Proof_V3.tex              # LaTeX paper
│   └── RH_Proof_V3.pdf              # Compiled PDF
├── code/
│   ├── RH_01_Quarter_Parameter.py   # 1/4 parameter verification
│   ├── RH_02_Smooth_Term.py         # Smooth term analysis
│   ├── RH_03_Trace_Matching.py      # Trace formula matching
│   ├── RH_04_Proof_Structure.py     # Proof structure verification
│   ├── RH_05_Spectral_Determinant.py
│   ├── RH_06_BC_Pole_Correspondence.py
│   ├── RH_07_Heat_Kernel.py
│   ├── RH_08_Spectral_Analysis.py
│   ├── RH_09_Smooth_Matching.py
│   ├── RH_10_Mellin_Transform.py
│   ├── RH_11_Eigenvalue_Simulation.py
│   ├── RH_12_Trace_Derivation.py
│   ├── RH_13_Z3_Verification.py
│   ├── RH_14_Formal_Verification.py
│   └── RH_15_Trace_Formula_Exactness.py
├── supplementary/
│   ├── RH_PROOF_RIGOROUS_COMPLETE.md
│   ├── TRACE_FORMULA_EXACTNESS_PROOF.md
│   └── SPECTRAL_DETERMINANT_XI_CONNECTION.md
└── README.md
```

---

## Proof Structure

| Step | Component | Method |
|------|-----------|--------|
| 1 | Operator definition | H = -i(q d/dq + 1/2) on L^2([0,1], Fisher weight) |
| 2 | Arc length | Fisher metric integral = pi (beta function identity) |
| 3 | Boundary condition | alpha = pi from geometric quantization |
| 4 | Amplitude derivation | Gutzwiller trace formula (monodromy matrix) |
| 5 | Trace formula | Term-by-term matching with Riemann-Weil |
| 6 | Spectral correspondence | Schwartz density + Riesz-Markov uniqueness |
| 7 | Self-adjointness | Spectral theorem implies real eigenvalues |

---

## Building the Paper

```bash
cd paper
pdflatex RH_Proof_V3.tex
pdflatex RH_Proof_V3.tex  # Run twice for references
```

---

## Running Verification Code

Requirements:
- Python 3.8+
- NumPy, SciPy, SymPy
- Z3 (z3-solver)

```bash
pip install numpy scipy sympy z3-solver
cd code
python RH_15_Trace_Formula_Exactness.py
```

---

## Citation

```bibtex
@article{Newton2026RH,
  author = {Newton, Mark},
  title = {Proof of the Riemann Hypothesis: Berry-Keating Spectral Approach with Fisher Information Metric},
  year = {2026},
  month = {February},
  doi = {10.5281/zenodo.18489099},
  url = {https://zenodo.org/records/18489099}
}
```

---

## License

CC BY 4.0

---

## Contact

For questions, open an issue on GitHub.
