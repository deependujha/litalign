# 📘 KL Divergence: Quick Notes

> **Kullback–Leibler (KL) divergence**

The **KL divergence** between two probability distributions $P$ and $Q$ is:

$$
D_{\text{KL}}(P || Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}
$$

In the context of RL:

- $P$: New (current) policy $\pi\_\theta$
- $Q$: Old (reference) policy $\pi\_{\theta\_{\text{old}}}$
- Used to **penalize divergence** from a reference policy

> - KL divergence yields how different two probability distributions are.
> - Or, how much information is lost when $Q$ is used to approximate $P$.

!!! danger "Note"
    KL divergence is **not symmetric**: $D\_{\text{KL}}(P || Q) \neq D\_{\text{KL}}(Q || P)$.
    It measures how much information is lost when using $Q$ to approximate $P$. And vice versa.

!!! warning "Important (GROK says:)"
    In the expression ( D\_{KL}(P||Q) ), the Kullback-Leibler (KL) divergence measures how much the probability distribution ( P ) diverges from the probability distribution ( Q ). Here's the breakdown:

    - **\( P \)**: This is the "true" or "target" distribution, the one you consider as the reference or the actual distribution you want to approximate.
    - **\( Q \)**: This is the "approximating" distribution, the one you use to estimate or approximate \( P \).

    ### Interpretation
    - \( D_{KL}(P||Q) \) quantifies the information loss when \( Q \) is used to approximate \( P \).
    - It is not symmetric, meaning \( D_{KL}(P||Q) \neq D_{KL}(Q||P) \), because the roles of the "true" and "approximating" distributions are not interchangeable.

    ### Formula
    The KL divergence is defined as:
    $$
    \[
    D_{KL}(P||Q) = \sum_x P(x) \log\left(\frac{P(x)}{Q(x)}\right)
    \]
    $$
    (for discrete distributions), or
    $$
    \[
    D_{KL}(P||Q) = \int P(x) \log\left(\frac{P(x)}{Q(x)}\right) dx
    \]
    $$
    (for continuous distributions).

    ### Key Points
    - \( P \) is the distribution you assume to be the true one.
    - \( Q \) is the distribution you use to model or estimate \( P \).
    - The asymmetry arises because \( P(x) \log\left(\frac{P(x)}{Q(x)}\right) \) weighs the log-ratio by \( P(x) \), not \( Q(x) \), so swapping them changes the result.

    So, in \( D_{KL}(P||Q) \), \( Q \) is used to estimate \( P \).


______________________________________________________________________

## ✅ PyTorch Implementation

Assuming your policies are represented as `Categorical` distributions (e.g., action logits):

- `kl_divergence.py`

```python
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

def kl_divergence_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """
    Computes KL divergence D_KL(P || Q) between two categorical distributions P and Q
    given their logits. Shape: [batch_size, num_actions]

    Args:
        logits_p: Logits of new/current policy πθ
        logits_q: Logits of old/reference policy πθ_old

    Returns:
        kl: Tensor of shape [batch_size] with KL divergence for each sample
    """
    p = F.log_softmax(logits_p, dim=-1)
    q = F.log_softmax(logits_q, dim=-1)

    p_prob = p.exp()
    kl = (p_prob * (p - q)).sum(dim=-1)
    return kl
```

______________________________________________________________________

## 🔍 Example Usage

```python
batch_size = 4
num_actions = 3

logits_new = torch.randn(batch_size, num_actions)
logits_old = logits_new + 0.1 * torch.randn(batch_size, num_actions)  # small shift

kl = kl_divergence_logits(logits_new, logits_old)
print("KL divergence per sample:", kl)
```

______________________________________________________________________

## ⚙️ For Gaussian (Continuous Action) Policies

If you're using a Gaussian policy (e.g., in continuous control with mean & std):

```python
from torch.distributions import Normal, kl_divergence

def kl_gaussian(mean_p, std_p, mean_q, std_q):
    dist_p = Normal(mean_p, std_p)
    dist_q = Normal(mean_q, std_q)
    return kl_divergence(dist_p, dist_q).sum(-1)  # Sum over action dims
```

______________________________________________________________________

## 🧠 Where You Use This

- PPO: `loss = policy_loss - β * KL(...)`
- DPO/GRPO: KL shows up in the policy regularizer
- TRPO: Uses KL as a trust region constraint
