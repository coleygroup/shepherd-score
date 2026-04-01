(theory)=
# Representations, Scoring, and Alignment

This page documents the mathematical formulations underlying ShEPhERD Score's interaction profile representations, similarity scoring functions, and analytical gradient derivations used during alignment.
Alignment has been driven by the automatic differentiation engines of Jax and PyTorch. However, since v1.3.1, analytical gradients have been implemented for PyTorch which significantly reduces memory overhead and speeds up computation; this is now the default path for PyTorch-based alignment in `MoleculePair`.
We use the notation provided in [Adams et al. (2025)](https://openreview.net/forum?id=KSLkFYHlYg).

## Base Representations

* **Shape:** Represented as a point cloud $\boldsymbol{x}_2 = \boldsymbol{S}_2 \in \mathbb{R}^{n_2\times3}$ (or atomic coordinates $\boldsymbol{C} \in \mathbb{R}^{n_1 \times 3}$).
* **Electrostatic Potential (ESP):** Represented as $\boldsymbol{x}_3 = (\boldsymbol{S}_3, \boldsymbol{v})$ where $\boldsymbol{S}_3 \in \mathbb{R}^{n_3 \times 3}$ is a surface point cloud and $\boldsymbol{v} \in \mathbb{R}^{n_3}$ is the Coulombic potential at each point. The potential $\boldsymbol{v}$ is defined by atomic partial charges $\boldsymbol{q} \in \mathbb{R}^{n_1}$ and positions $\boldsymbol{r}$:

$$\boldsymbol{v}[k] = \frac{1}{4 \pi \epsilon_0} \sum^{n_3}_{j=1} \frac{q[k]}{\|\boldsymbol{r}[k] - \boldsymbol{r}[j] \|^2}$$

* **Pharmacophores:** Represented as $\boldsymbol{x}_4 = (\boldsymbol{p}, \boldsymbol{P}, \boldsymbol{V})$.
  * $\boldsymbol{p} \in \mathbb{R}^{n_4 \times N_p}$: one-hot encodings of $N_p$ types.
  * $\boldsymbol{P} \in \mathbb{R}^{n_4 \times 3}$: 3D coordinates.
  * $\boldsymbol{V} \in \{ \mathbb{S}^2, \boldsymbol{0}\}^{n_4}$: relative unit vectors ($|\boldsymbol{V}[k]| = 1$) for directional pharmacophores, or zero vectors ($|\boldsymbol{V}[k]| = 0$) for directionless ones.

## General Similarity and Alignment

For any two point clouds $\boldsymbol{Q}_A$ and $\boldsymbol{Q}_B$ with points $\boldsymbol{r}_k$, the first-order Gaussian overlap is:

$$O_{A,B} = \sum_{a \in \boldsymbol{Q}_A} \sum_{b \in \boldsymbol{Q}_B} w_{a,b} \left(\frac{\pi}{2 \alpha}\right)^{\frac{3}{2}} \exp{\left(-\frac{\alpha}{2}\|\boldsymbol{r}_a - \boldsymbol{r}_b\|^2\right)}$$

The Tanimoto similarity function is:

$$\text{sim}^{*}(\boldsymbol{Q}_A, \boldsymbol{Q}_B) = \frac{O_{A,B}}{O_{A,A} + O_{B,B} - O_{A,B}}$$

The optimal alignment objective is:

$$\text{sim}(\boldsymbol{Q}_A, \boldsymbol{Q}_B) = \max_{\boldsymbol{R, t}}{\text{sim}^{*} (\boldsymbol{R}\boldsymbol{Q}_A^T + \boldsymbol{t}, \boldsymbol{Q}_B)}$$

where $\boldsymbol{R} \in SO(3)$ and $\boldsymbol{t}\in T(3)$.

## Specific Scoring Functions

The variables $w_{a,b}$ and $\alpha$ from the general overlap function adapt based on the representation being scored:

**Shape Scoring**

* *Volumetric (using atoms $\boldsymbol{C}$):* $w_{a,b} = 2.7$, $\alpha = 0.81$.
* *Surface (using points $\boldsymbol{S}$):* $w_{a,b} = 1$, $\alpha = \Psi(n_2)$ (a predefined constant based on $n_2$).

**ESP Scoring**

The overlap function incorporates the potential difference:

$$O_{A,B}^{\text{ESP}} = \sum_{a \in \boldsymbol{Q}_A} \sum_{b \in \boldsymbol{Q}_B} \left(\frac{\pi}{2 \alpha}\right)^{\frac{3}{2}} \exp{\left(-\frac{\alpha}{2}\|\boldsymbol{r}_a - \boldsymbol{r}_b\|^2\right)} \exp{\left(-\frac{\|\boldsymbol{v}_{A}[a] - \boldsymbol{v}_{B}[b]\|^2}{\lambda}\right)}$$

where $\alpha = \Psi(n_3)$ and $\lambda = \frac{0.3}{(4 \pi \epsilon_0)^2}$.

**Pharmacophore Scoring**

Calculated per pharmacophore type $m \in \mathcal{M}$ (where $|\mathcal{M}| = N_p$). The overlap for a specific type $m$ is:

$$O^\text{pharm}_{A,B;m} = \sum_{a \in \boldsymbol{Q}_{A,m}} \sum_{b \in \boldsymbol{Q}_{B,m}} w_{a,b;m}\left(\frac{\pi}{2 \alpha_m}\right)^{\frac{3}{2}} \exp{\left(-\frac{\alpha_m}{2}\|\boldsymbol{r}_a - \boldsymbol{r}_b\|^2\right)}$$

The total similarity sums the overlaps across all types:

$$\text{sim}_{\text{pharm}}^{*}(\boldsymbol{x}_{4,A}, \boldsymbol{x}_{4,B}) = \frac{\sum_{m\in \mathcal{M}} O_{A,B; m}}{\sum_{m \in \mathcal{M}} O_{A,A; m} + O_{B,B; m} - O_{A,B; m}}$$

The vector weighting $w_{a,b;m}$ depends on directionality ($\alpha_m$ is a constant per type):

$$w_{a,b;m} = \begin{cases} 1 & \text{if } m \text{ is non-directional}, \\ \frac{\boldsymbol{V}[a]_{m}^\top \boldsymbol{V}[b]_{m} + 2}{3} & \text{if } m \text{ is directional}. \end{cases}$$

*(Note: For aromatic rings, the absolute value of the dot product $|\boldsymbol{V}[a]_{m}^\top \boldsymbol{V}[b]_{m}|$ is used.)*

---

## Analytical Gradients

Analytical gradients have been implemented for **pharmacophore**, **shape**, and
**shape-with-avoid** alignment, replacing PyTorch autograd. All implementations
live in `shepherd_score/score/analytical_gradients/` (PyTorch in `_torch.py`,
re-exported via `__init__.py`) and are called from the optimizer loops in
`shepherd_score/alignment/_torch_analytical.py`. The resulting speedup is
approximately 2–2.5× over autograd.

The SE(3) parameter vector is always `(q_w, q_x, q_y, q_z, t_x, t_y, t_z)` —
4 quaternion components followed by 3 translation components. All functions
support both single `(7,)` and batched `(B,7)` inputs.

### 1. Tanimoto Chain Rule

The self-overlaps $O_{A,A}$ and $O_{B,B}$ are **invariant** to any rigid SE(3)
transformation. Defining $U = O_{A,A} + O_{B,B}$, the Tanimoto gradient
simplifies via the quotient rule:

$$\nabla_{\theta} S = \frac{U}{(U - O_{A,B})^2} \nabla_{\theta} O_{A,B}$$

$U$ is computed once before the optimization loop. Only $\nabla_{\theta} O_{A,B}$
needs to be evaluated each step. The loss is $1 - S$.

### 1b. Tversky Chain Rule (pharmacophore only)

For Tversky similarity, $D = \sigma \cdot O_{A,A} + (1-\sigma) \cdot O_{B,B}$
is a **constant** w.r.t. SE(3). The similarity is:

$$\text{sim}^{*}(\boldsymbol{Q}_A, \boldsymbol{Q}_B) = \min\left(\frac{O_{A,B}}{D}, 1.0\right)$$

The gradient scales directly by a constant factor (no quotient rule):

$$\nabla_{\theta} S = -\frac{\mathbf{1}[S < 1.0]}{D} \nabla_{\theta} O_{A,B}$$

where the indicator function $\mathbf{1}[S < 1.0]$ is 1 when not clamped, 0
otherwise. Sigma variants: `'tversky'`→0.95, `'tversky_ref'`→1.0,
`'tversky_fit'`→0.05.

### 2. Shape and ESP Gradients

Let $\boldsymbol{\Delta}_{ab} = \boldsymbol{R}\boldsymbol{r}_a + \boldsymbol{t} - \boldsymbol{r}_b$ and $E_{ab} = \exp(-\frac{\alpha}{2}\|\boldsymbol{\Delta}_{ab}\|^2)$.

For **shape**, the pair constant is $C_{ab} = w_{ab} \left(\frac{\pi}{2\alpha}\right)^{3/2}$ (with $w_{ab}=1$ for surface, $w_{ab}=2.7$ for volumetric).
For **ESP**, the pair constant absorbs the potential difference: $\tilde{C}_{ab} = \left(\frac{\pi}{2\alpha}\right)^{3/2} \exp\!\left(-\|\boldsymbol{v}_A[a]-\boldsymbol{v}_B[b]\|^2 / \lambda\right)$. The ESP potential is invariant to rigid motion, so $\tilde{C}_{ab}$ is a pure constant during alignment.

Both cases share the same gradient structure (substituting $C_{ab}$ or $\tilde{C}_{ab}$):

$$\nabla_{\boldsymbol{t}} O_{A,B} = \sum_{a,b} -\alpha\, C_{ab}\, E_{ab}\, \boldsymbol{\Delta}_{ab}$$

$$\nabla_{\boldsymbol{R}} O_{A,B} = \sum_{a,b} -\alpha\, C_{ab}\, E_{ab}\, (\boldsymbol{\Delta}_{ab}\, \boldsymbol{r}_a^\top)$$

### 3. Pharmacophore Gradients

The pairwise term per pharmacophore type $m$ is $f_{ab} = w'_{ab;m}\, K_m\, E_{ab}$ where $K_m = (\pi/(2\alpha_m))^{3/2}$ and $E_{ab} = \exp(-\frac{\alpha_m}{2}\|\boldsymbol{P}'_a - \boldsymbol{P}_b\|^2)$.

**Translation gradient** (weight $w'$ is rotation-only, so only $E_{ab}$ contributes):

$$\nabla_{\boldsymbol{t}} O_{A,B} = \sum_{m} \sum_{a,b} -\alpha_m\, w'_{ab;m}\, K_m\, E_{ab}\, (\boldsymbol{P}'_a - \boldsymbol{P}_b)$$

**Rotation matrix gradient** (product rule over spatial and weight terms):

$$\nabla_{\boldsymbol{R}} O_{A,B} = \sum_{m} \sum_{a,b} K_m E_{ab} \left[ \nabla_{\boldsymbol{R}} w'_{ab;m} - \alpha_m\, w'_{ab;m}\, (\boldsymbol{P}'_a - \boldsymbol{P}_b)\boldsymbol{P}_a^\top \right]$$

The directional weight gradients are:

* **Non-directional** (Hydrophobe, ZnBinder, Anion, Cation): $\nabla_{\boldsymbol{R}} w' = \mathbf{0}$
* **Directional** (Acceptor, Donor, Halogen): $w = (D_{ab}+2)/3 \implies \nabla_{\boldsymbol{R}} w' = \tfrac{1}{3}\boldsymbol{V}_b\boldsymbol{V}_a^\top$
* **Aromatic**: $w = (|D_{ab}|+2)/3 \implies \nabla_{\boldsymbol{R}} w' = \tfrac{1}{3}\operatorname{sgn}(D_{ab})\boldsymbol{V}_b\boldsymbol{V}_a^\top$

where $D_{ab} = (\boldsymbol{R}\boldsymbol{V}_a)^\top \boldsymbol{V}_b$.

### 4. Avoid Penalty Gradients

The linear hard-sphere avoid penalty is:

$$A = \sum_{a \in \text{fit\_avoid}} \sum_{b \in \text{avoid}} \operatorname{relu}\!\left(\frac{d_0 - \|\boldsymbol{P}'_a - \boldsymbol{P}_b\|}{d_0}\right)$$

where $d_0$ is `avoid_min_dist`. The gradient is active only where $0 < \|\boldsymbol{\Delta}_{ab}\| < d_0$:

$$\frac{\partial A}{\partial \boldsymbol{P}'_a} = -\frac{1}{d_0} \sum_b \mathbf{1}[0 < \|\boldsymbol{\Delta}_{ab}\| < d_0]\, \hat{\boldsymbol{\Delta}}_{ab}$$

$$\nabla_{\boldsymbol{t}} A = \sum_a \frac{\partial A}{\partial \boldsymbol{P}'_a}, \qquad \nabla_{\boldsymbol{R}} A = \left(\frac{\partial A}{\partial \boldsymbol{P}'}\right)^\top \boldsymbol{P}_{\text{orig}}$$

The combined loss for shape-with-avoid is $\mathcal{L} = (1 - S_\text{shape}) + \lambda_\text{avoid} \cdot A$. Gradients add linearly before quaternion projection.

### 5. Quaternion Projection

The $3\times3$ rotation gradient $\boldsymbol{G}$ is projected onto quaternion parameters via the Frobenius inner product with the quaternion Jacobians:

$$\frac{\partial \mathcal{L}}{\partial q_k} = \operatorname{Tr}\!\left(\boldsymbol{G}^\top \frac{\partial \boldsymbol{R}}{\partial q_k}\right)$$

The four Jacobians $\partial\boldsymbol{R}/\partial q_k$ are linear functions of $\boldsymbol{q}$, e.g.:

$$\frac{\partial \boldsymbol{R}}{\partial q_w} = \begin{bmatrix} 0 & -2q_z & 2q_y \\ 2q_z & 0 & -2q_x \\ -2q_y & 2q_x & 0 \end{bmatrix}$$

Because the optimizer works with unnormalized raw quaternions $\hat{\boldsymbol{q}}$, the chain rule for normalization $\boldsymbol{q} = \hat{\boldsymbol{q}}/\|\hat{\boldsymbol{q}}\|$ is applied:

$$\nabla_{\hat{\boldsymbol{q}}} = \frac{1}{\|\hat{\boldsymbol{q}}\|}\left(\boldsymbol{I} - \boldsymbol{q}\boldsymbol{q}^\top\right)\nabla_{\boldsymbol{q}}$$

---


### Key constants — `score/constants.py`

* `P_ALPHAS`: per-type $\alpha_m$ (all 1.0 except aromatic/hydrophobe = 0.7)
* `P_TYPES`: `('Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'Halogen', 'Cation', 'Anion', 'ZnBinder', 'Dummy')`


### References
```{admonition} Citation
:class: note

If you use or adapt this work, please cite:

Adams, K., Abeywardane, K., Fromer, J., & Coley, C. W. (2025). ShEPhERD: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design. *The Thirteenth International Conference on Learning Representations*. https://openreview.net/forum?id=KSLkFYHlYg
```
