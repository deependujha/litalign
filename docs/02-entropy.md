# 🧠 Entropy in Reinforcement Learning

### 🚀 Intuition

- **Probability** tells us how *certain* an agent is about taking an action.

- **Surprise** captures *how unexpected* an outcome is.

- So, a natural idea is:

  $$
  \text{Surprise}(a) = \frac{1}{P(a)}
  $$

But this isn't ideal:

- If $P(a) = 1$, surprise should be 0 — but $\frac{1}{1} = 1$, which doesn’t work.
- Instead, we define surprise using logarithms:

______________________________________________________________________

### 📐 Surprise = Log Inverse Probability

$$
\text{Surprise}(a) = \log\left(\frac{1}{P(a)}\right) = -\log P(a)
$$

So, the less likely the action, the greater the surprise.

______________________________________________________________________

### 📊 Entropy = Expected Surprise

Entropy is the **expected surprise over all possible actions**:

$$
\begin{align*}
\text{Entropy}(\pi) &= \sum_{a \in \mathcal{A}} P(a) \cdot \text{Surprise}(a) \\
&= \sum_{a \in \mathcal{A}} P(a) \cdot (-\log P(a)) \\
&= -\sum_{a \in \mathcal{A}} P(a) \cdot \log P(a)
\end{align*}
$$

______________________________________________________________________

### 🔁 What Entropy Tells Us

| Distribution       | Entropy | Notes                        |
| ------------------ | ------- | ---------------------------- |
| [1.0, 0.0, 0.0]    | 0       | Fully deterministic          |
| [0.7, 0.2, 0.1]    | Low     | Fairly confident             |
| [0.33, 0.33, 0.34] | High    | Very uncertain (max entropy) |

> 📌 Entropy is **highest** when all actions are equally likely (pure exploration),
> and **lowest** when the policy is deterministic (pure exploitation).

______________________________________________________________________

### 🧪 PyTorch: Compute Entropy

If you have a probability distribution (e.g. from `softmax`), you can compute entropy like this:

```python
import torch
import torch.nn.functional as F

# Example: logits for 3 actions
logits = torch.tensor([1.0, 0.5, -0.5])

# Get action probabilities
probs = F.softmax(logits, dim=-1)

# Compute entropy
entropy = -torch.sum(probs * torch.log(probs + 1e-8))  # +1e-8 for numerical stability

print("Entropy:", entropy.item())
```

- or, using `Categorical` distribution:

```python
import torch

# Example logits for a single state with 3 actions
logits = torch.tensor([1.0, 0.5, -0.5])

# Create a categorical distribution
dist = torch.distributions.Categorical(logits=logits)

# Compute entropy
entropy = dist.entropy()

print("Entropy:", entropy.item())
```

______________________________________________________________________

### ⚙️ When is Entropy Used in RL?

- **Policy Gradient Methods (PPO, A2C, etc.):**

  - Add **entropy bonus** to the loss:

    ```python
    total_loss = ppo_loss - entropy_coeff * entropy
    ```

  - Prevents policy from collapsing too early into deterministic behavior.

  - Encourages *ongoing exploration* especially in early training.

- **Entropy Coefficient (hyperparameter):**

  - Typically a small value (e.g. `0.01`, `0.001`)
  - Can be annealed (decayed) over time.

______________________________________________________________________

### 🧠 Summary

- Entropy is a measure of **uncertainty** in the policy.
- Encouraging entropy helps with **exploration** in RL.
- PPO uses an **entropy bonus** to maintain a balance between exploring and exploiting.
