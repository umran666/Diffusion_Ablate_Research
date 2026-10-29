# Contributing to the Diffusion Ablation Research Group

Welcome to our research collaboration repository! We appreciate your insights, mathematical proofs, experimental results, and architectural extensions.

---

## 🎯 Contribution Areas

1. **Theoretical Derivations & Proofs**:
   - Refinements to existing theorems (e.g., tighter bounds on trajectory KL divergence, generalized multi-modal cross-talk bounds).
   - Novel proofs for unaddressed architectures (e.g., discrete diffusion, autoregressive hybrid models).

2. **Empirical Benchmarks & Hardware Findings**:
   - Reproducibility logs and metric tables from runs on cloud infrastructure (e.g., Modal, AWS, GCP, RunPod).
   - Analysis of benign fidelity degradation vs. refusal suppression across various threshold hyperparameters.

3. **Taxonomy & Architecture Specifications**:
   - Updating architectural intervention sites as new open-weight DiT foundation models are released.

---

## 🛠️ Step-by-Step Workflow

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/umran666/Diffusion_Ablate_Research.git
   cd Diffusion_Ablate_Research
   ```

2. **Create a Dedicated Branch**:
   ```bash
   git checkout -b research/your-topic-name
   ```

3. **Edit `research_finds.md`**:
   - Make your additions, corrections, or notes directly in `research_finds.md`.
   - Maintain clear LaTeX math formatting and section numbering.

4. **Synchronize & Verify**:
   - Run the included zero-dependency Python script:
     ```bash
     python verify_integrity.py --sync
     ```
   - This ensures `research_finds` (the plaintext twin) is automatically updated and checks that all LaTeX blocks and section headers are intact.

5. **Commit and Push**:
   ```bash
   git add research_finds.md research_finds
   git commit -m "docs(secX): add proof for Theorem X.Y"
   git push origin research/your-topic-name
   ```

6. **Submit a Pull Request**:
   - Open a PR against `main`.
   - In the PR description, summarize:
     - What section was modified/added.
     - The mathematical or empirical basis for the change.
     - Any relevant references or paper citations.

---

## ✅ Pull Request Checklist

Before submitting, please ensure:
- [ ] No large files or model checkpoints (`.safetensors`, `.pt`, `.bin`) are included.
- [ ] No private tokens (`HF_TOKEN`, cloud credentials) are included in text or diffs.
- [ ] `python verify_integrity.py` exits with status `0` (SUCCESS).
- [ ] Any new theorem follows the standard format: **Theorem X.Y (Name)**, Assumptions, Statement, and Proof.

Thank you for collaborating!
