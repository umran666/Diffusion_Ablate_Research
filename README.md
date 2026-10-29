# Mathematical Foundations & Empirical Findings in Diffusion Representation Ablation

> **Private Collaborative Research Repository**  
> *Targeted Subspace Surgery, Formal Generalization Bounds, and Live Empirical Validation across DiT Architectures (FLUX.1, SD3.5, Ideogram 4, MM-DiT, and Video DiTs)*

---

## 📌 Overview

This repository hosts the primary scientific manuscript, formal mathematical derivations, and empirical benchmark findings for representation ablation in multi-modal diffusion transformers (MM-DiTs) and rectified flow models.

The primary document is [`research_finds.md`](./research_finds.md), with an identical plaintext mirror [`research_finds`](./research_finds). It contains 24 structured sections covering theoretical foundations, closed-form projections, adversarial hardening, video temporal dynamics, and real-world hardware verification.

---

## 🔬 Key Research Themes & Section Index

| Section | Focus Area | Key Theoretical & Empirical Results |
|:---:|:---|:---|
| **1–5** | Foundational Diffusion Ablation | Linear representation hypothesis, concept subspace extraction, rank-$k$ SVD projections. |
| **6** | Trajectory KL Divergence | Theorem 6.1: Formal KL divergence bound along probability flow ODE trajectories. |
| **7** | Multi-Modal DiT Cross-Talk | Mathematical analysis of joint text-image stream cross-talk and concept leakage. |
| **8** | Adversarial Hardening | Proof of circumvention vulnerability and Mahalanobis whitening defenses. |
| **9** | Spectral Pareto Frontier | Optimal trade-off between target concept suppression and benign representation preservation. |
| **10–16** | Multi-Architecture Generalization | Cross-validation on PixArt-$\alpha$, FLUX.1 Schnell/Dev, and Stable Diffusion 3.5 Large. |
| **17** | Joint Multi-Concept Orthogonalization | Extension of single-concept projection to non-orthogonal multi-concept manifolds. |
| **18** | Closed-Form Subspace Surgery | Theorems 18.1–18.4: Exact solutions for simultaneous multi-concept null-space projectors. |
| **19** | Whitened Hardening Proofs | Theorem 19.1: Invariant subspace protection against adversarial feature rotations. |
| **20** | Continuous-Time Flow Matching | Integration of Riemannian geometry and flow vector fields under representation knockout. |
| **21** | Cross-Attention vs. FFN Surgery | Analytical comparison of intervention loci: why feed-forward layers govern semantic retrieval. |
| **22** | Discrete State-Space & Masked DiT | Extension to discrete and categorical token-based diffusion mechanisms. |
| **23** | Video DiT Spatiotemporal Surgery | Theorems 23.1–23.6: Temporal attention disentanglement, 3D causal convolutions, and flicker bounds. |
| **24** | Empirical Live Modal.com Validation | **Live A100-80GB Empirical Run**: Ideogram 4 FP8 gateway weight ablation ($\|W^* v^\top\|_2 = 0.021006$, 99.23% refusal suppression, 98.11% benign fidelity). |

---

## 🤝 Collaborative Guidelines for Co-Researchers

We welcome contributions, mathematical refinements, empirical benchmarks, and architectural extensions from invited collaborators.

### 1. Dual-File Synchronicity Rule
To ensure compatibility across different viewing tools and academic parsing pipelines, **`research_finds.md`** and **`research_finds`** are maintained as exact byte twins.
Whenever you make changes to `research_finds.md`, you can update the mirror or run the included verification script:
```bash
# Verify integrity and sync twin
python verify_integrity.py --sync
```

### 2. Verification Before Committing
Run the integrity check to verify that:
- All 24 section headers are preserved.
- LaTeX delimiters (`$...$` and `$$...$$`) are properly closed.
- Both files match bit-for-bit.
```bash
python verify_integrity.py
```

### 3. Git Workflow
- Please create a feature branch for substantive edits or new sections:
  ```bash
  git checkout -b feature/refine-section-X
  # make edits to research_finds.md
  python verify_integrity.py --sync
  git commit -am "docs: refine theorem X proof in section Y"
  git push origin feature/refine-section-X
  ```
- Open a Pull Request for discussion among the research group.

### 4. Mathematical Formatting Conventions
- Multi-term equations, bounds, and matrices must be placed in display math blocks (`$$...$$`).
- Single variables and short expressions should use inline math (`$...$`).
- State theorems formally using bold tags (**Theorem X.Y (Name)**) followed by assumptions, proof, and asymptotic bounds.

---

## 🛡️ Repository Constraints

> [!IMPORTANT]
> - **No Model Checkpoints**: Do not commit model weights (`.safetensors`, `.pt`, `.bin`) or media outputs (`.png`, `.jpg`, `.mp4`). All weights and ablation artifacts are managed on cloud storage/volumes.
> - **Strict Confidentiality**: This repository is strictly private for authorized collaborators. Do not fork or push to public remotes without prior consensus.

---

## 📜 Citation

If you use these findings or derivations in subsequent papers or drafts, please cite:
```bibtex
@misc{umran2026diffusion_ablate,
  author = {Umran, Shaik and Collaborators},
  title = {Mathematical Foundations, Geometric Subspace Surgery, and Empirical Verification for Representation Ablation in Diffusion and Flow-Matching DiTs},
  year = {2026},
  publisher = {Private Research Group},
  howpublished = {Internal Research Repository}
}
```
