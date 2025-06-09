# Direct Preference Optimization (DPO)

> Your language model is secretly a Reward Model.

## Formula:

$$
\\mathcal{L}_{DPO} = - \\mathbb{E}_{x, y_w, y_l} \\left[ \\log \\sigma \\left(\\beta \\log \\frac{\\pi\_\\theta(y_w|x)}{\\pi\_{ref}(y_w|x)} - \\beta \\log \\frac{\\pi\_\\theta(y_l|x)}{\\pi\_{ref}(y_l|x)} \\right)\\right]
$$

the above loss function when being implemented replaces `division with subtraction`:

$$
\\mathcal{L}_{DPO} = - \\mathbb{E}_{x, y_w, y_l} \\left[ \\log \\sigma \\left( \\beta (\\log \\pi\_\\theta(y_w|x) - \\log \\pi\_{ref}(y_w|x)) - \\beta (\\log \\pi\_\\theta(y_l|x) - \\log \\pi\_{ref}(y_l|x)) \\right)\\right]
$$

### 😉 log sigmoid

$$ LogSigmoid (X) = \\log \\sigma(x) = \\log \\left( \\frac{1}{1 + e^{-x}} \\right ) $$

______________________________________________________________________

## Code

```python
import torch.nn.functional as F

def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
    """
    pi_logps: policy logprobs, shape (B,)
    ref_logps: reference model logprobs, shape (B,)
    yw_idxs: preferred completion indices in [0, B-1], shape (T,)
    yl_idxs: dispreferred completion indices in [0, B-1], shape (T,)
    beta: temperature controlling strength of KL penalty
    Each pair of (yw_idxs[i], yl_idxs[i]) represents the
    indices of a single preference pair.
    """
    pi_yw_logps, pi_yl_logps = pi_logps[yw_idxs], pi_logps[yl_idxs]
    ref_yw_logps, ref_yl_logps = ref_logps[yw_idxs], ref_logps[yl_idxs]
    pi_logratios = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps
    losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
    rewards = beta * (pi_logps - ref_logps).detach()
    return losses, rewards
```

______________________________________________________________________

## 🧠 Direct Preference Optimization (DPO)

DPO is a method to align language models **without reinforcement learning** or training a reward model.
Instead of generating a numeric reward signal, we use **human preferences** directly.

______________________________________________________________________

## 🪢 Setup

We are given a dataset of:

```
(prompt, accepted_response, rejected_response)
=> (x, y_w, y_l)
```

The goal is to fine-tune a policy (language model) to prefer `y_w` over `y_l`.

______________________________________________________________________

## 🏗️ Core Idea

Instead of RL or reward modeling, DPO fine-tunes a model to maximize the **preference likelihood**:

> The model should assign **higher probability** to `y_w` than to `y_l`.

We use a **reference model** (`π_ref`) — typically the base model — which is **frozen**.
We compare the fine-tuned model (`π_θ`) against this.

______________________________________________________________________

## 🔍 Log Probability Computation

For each prompt-response pair:

```python
log_pi_theta_yw = log π_θ(y_w | x)   # logprob of accepted response
log_pi_theta_yl = log π_θ(y_l | x)   # logprob of rejected response

log_pi_ref_yw = log π_ref(y_w | x)
log_pi_ref_yl = log π_ref(y_l | x)
```

We usually compute `log π(y|x)` by **summing the log probabilities** of each token in the response.

______________________________________________________________________

## 🧮 Loss Function

DPO loss:

$$
\\mathcal{L}_{DPO} = - \\mathbb{E}_{x, y_w, y_l} \\left[ \\log \\sigma \\left( \\beta \\cdot \\left( \\log \\frac{\\pi\_\\theta(y_w|x)}{\\pi\_\\text{ref}(y_w|x)} - \\log \\frac{\\pi\_\\theta(y_l|x)}{\\pi\_\\text{ref}(y_l|x)} \\right) \\right) \\right]
$$

Using log identities:

$$
= - \\log \\sigma \\left( \\beta \\cdot \\left[ (\\log π_θ(y_w) - \\log π_θ(y_l)) - (\\log π_ref(y_w) - \\log π_ref(y_l)) \\right] \\right)
$$

______________________________________________________________________

## 📦 Why LogSigmoid?

We use:

$$
\\log \\sigma(z) = \\log \\left( \\frac{1}{1 + e^{-z}} \\right)
$$

### Intuition:

- It acts like a **binary preference classifier**.
- Encourages model to **rank `y_w` over `y_l`**.
- Smooth, differentiable, and **stable even for large `z`**.
- Equivalent to maximizing the **log-likelihood of choosing the better output**.

When:

- `Δ ≫ 0`: → `logsigmoid` ≈ 0 → ✅ low loss (model prefers `y_w`)
- `Δ ≪ 0`: → `logsigmoid` ≪ 0 → ❌ high loss (model prefers `y_l`)

______________________________________________________________________

## 💻 Code

```python
import torch.nn.functional as F

def dpo_loss(pi_logps, ref_logps, yw_idxs, yl_idxs, beta):
    """
    pi_logps: model logprobs, shape (B,)
    ref_logps: reference model logprobs, shape (B,)
    yw_idxs, yl_idxs: indices for chosen and rejected responses
    beta: scaling factor (temperature)
    """
    pi_yw_logps = pi_logps[yw_idxs]
    pi_yl_logps = pi_logps[yl_idxs]
    ref_yw_logps = ref_logps[yw_idxs]
    ref_yl_logps = ref_logps[yl_idxs]

    pi_logratios = pi_yw_logps - pi_yl_logps
    ref_logratios = ref_yw_logps - ref_yl_logps

    loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios))

    # Optional: reward signal for logging or auxiliary losses
    rewards = beta * (pi_logps - ref_logps).detach()

    return loss, rewards
```

______________________________________________________________________

## ⚠️ Notes

- DPO **doesn't use a separate reward model**.
- No explicit KL-divergence term like in PPO — but the **log-ratio implicitly acts like a KL regularizer**.
- You typically **sum logprobs over the entire sequence** to get `log π(y | x)`.
- DPO is **much easier to train** than PPO, and doesn’t need rollouts or reward shaping.

______________________________________________________________________

## 🔄 Variants / Flexibility

- Some variants skip the `π_ref` term, especially when no base model is available.
- `β` controls the **strength of alignment** — higher β pushes stronger preference.

______________________________________________________________________

## 🏁 Final Summary

- DPO = simple, stable, and effective fine-tuning method using preference data.
- Just needs a frozen base model and a set of chosen vs rejected completions.
- It's a **clean way to align LLMs** without the full complexity of RLHF.

______________________________________________________________________

## How is language model aligned?

- We input prompt into the model, and it yields the probability distribution over the next token.
- For each accepted and rejected response, we get the probability of the next token and take its log.
- We then input `prompt+next token` into the model, and repeat the process, to get the probability distribution over the next token.
- We continue this until we reach the end of the response.
- Finally, we sum the log probabilities of all tokens in the response to get the log probability of the entire response to receive `log π(y | x)`.

!!! bug "we sum the probability of next tokens and then take the log, we take the log of each next token probability and then sum?"

```
In DPO (and most language modeling tasks), we **take the log of each token's probability and then sum** them.

### Why?

Because the probability of the entire sequence is the **product** of individual token probabilities:

$$
P(y \mid x) = P(y_1 \mid x) \cdot P(y_2 \mid x, y_1) \cdot P(y_3 \mid x, y_1, y_2) \cdots
$$

Taking the log:

$$
\log P(y \mid x) = \log P(y_1 \mid x) + \log P(y_2 \mid x, y_1) + \log P(y_3 \mid x, y_1, y_2) + \cdots
$$

So, you sum **log-probabilities per token** to get the **sequence-level log-probability**.
```
