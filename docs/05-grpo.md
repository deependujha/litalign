# 🧠 GRPO: Group Relative Policy Optimization

!!! example "GRPO Overview"
   my understanding of grpo is that it's very similar to ppo. Some changes are: it reintroduces explicit kl divergence term for loss, and replaces value model and instead rather than generating one response, it uses beam search to get K responses, and the group of K response is ranked relatively
   - advantage of taken action a_i = (reward for a_i - mean reward for all the action(output))/std dev

   rest all seems similar.


## 🔍 What is GRPO?

**GRPO (Group Relative Policy Optimization)** is an RLHF (Reinforcement Learning from Human Feedback) method inspired by PPO (Proximal Policy Optimization), designed to fine-tune large language models (LLMs) using relative ranking feedback **within a group of outputs**, instead of absolute scores or pairwise preferences.

Unlike PPO:

- GRPO **does not use a value model**.
- It **uses beam search** to generate multiple completions (responses) per prompt.
- It calculates **relative advantages** within each group of completions.
- It often **includes an explicit KL divergence term** in the loss.

______________________________________________________________________

## 🏗️ GRPO Training Pipeline

### 1. **Pretrained Reward Model**

- Train a reward model from preference data (like in DPO/PPO).
- Reward model maps a (prompt, completion) pair to a scalar score.

### 2. **Prompt → Beam Search → K Completions**

- For each prompt, run **beam search** to generate **K candidate completions**.
- These K completions form a **group**.

### 3. **Score Group Completions**

- For each generated response $y_i$ in group $G$, compute:

  $$
  R(y_i) = \\text{reward model output}
  $$

### 4. **Groupwise Advantage Calculation**

For each response in group $G$:

$$
A_i = \\frac{R(y_i) - \\mu_G}{\\sigma_G}
$$

Where:

- $\\mu_G$ is the **mean** reward in group $G$
- $\\sigma_G$ is the **standard deviation**

This normalizes rewards and allows the model to focus on **relative improvement**.

### 5. **PPO-style Policy Update**

Use PPO’s clipped surrogate objective:

$$
L^{CLIP}(\\theta) = \\mathbb{E} \\left[ \\min \\left( r\_\\theta A_i, \\text{clip}(r\_\\theta, 1-\\epsilon, 1+\\epsilon) A_i \\right) \\right]
$$

Where:

- $r\_\\theta = \\frac{\\pi\_\\theta(y_i)}{\\pi\_{\\text{old}}(y_i)}$
- $\\pi\_\\theta$: updated policy
- $\\pi\_{\\text{old}}$: policy before update

### 6. **Optional KL Penalty**

Some GRPO variants include an **explicit KL divergence penalty**:

$$
L\_{KL} = \\beta \\cdot \\text{KL}[\\pi\_\\theta || \\pi\_{\\text{ref}}]
$$

Total Loss:

$$
L\_{total} = L^{CLIP} + L\_{KL}
$$

______________________________________________________________________

## Loss

GRPO operates **over groups of responses** (like beams from beam search), so the loss is **summed over all elements in the group**.

Here’s how it works conceptually:

______________________________________________________________________

### ✅ Group-wise Computation in GRPO

For each prompt $x$, you generate a group of responses ${y_1, y_2, ..., y_K}$ using beam search.

Then:

1. **Get rewards**:
   Use a reward model to get scores ${r_1, r_2, ..., r_K}$ for each $y_i$

2. **Normalize rewards within the group**:
   This gives you relative advantages:

$$
A_i = \\frac{r_i - \\mu_r}{\\sigma_r + \\epsilon}
$$

3. **Compute log-probabilities**:

   - $\\log \\pi\_\\theta(y_i|x)$: current policy
   - $\\log \\pi\_{\\text{ref}}(y_i|x)$: reference policy (often frozen)

4. **Compute surrogate objective**:
   For each $y_i$ in the group, compute the PPO-style clipped term with the normalized reward as the advantage:

$$
\\text{loss}_i = - \\min \\left( r_i \\cdot \\frac{\\pi_\\theta(y_i|x)}{\\pi\_{\\text{ref}}(y_i|x)}, \\text{clip}\\left(\\frac{\\pi\_\\theta(y_i|x)}{\\pi\_{\\text{ref}}(y_i|x)}, 1 - \\epsilon, 1 + \\epsilon \\right) \\cdot r_i \\right)
$$

5. **Sum over the group**:
   The total loss for one prompt is:

$$
\\mathcal{L}_{\\text{GRPO}}^{(x)} = \\sum_{i=1}^{K} \\text{loss}\_i
$$

6. **Average over batch**:
   Final training loss is averaged across all prompts in the batch.

______________________________________________________________________

So yes — **GRPO loss aggregates over the group** for each prompt to respect relative rankings rather than treating responses independently. This is what gives it the “group relative” flavor, unlike PPO which considers each sample in isolation.

______________________________________________________________________

## 🧪 Key Differences from PPO

| Aspect             | PPO                             | GRPO                          |
| ------------------ | ------------------------------- | ----------------------------- |
| Value Model        | Required                        | Not used                      |
| Response Sampling  | 1 response per prompt           | K responses via beam search   |
| Advantage Estimate | GAE or MC return                | Relative group-wise advantage |
| Reward Source      | Return or reward model          | Reward model (pretrained)     |
| KL Term            | Optional (implicit or explicit) | Often explicitly added        |

______________________________________________________________________

## 🧠 Why Group Normalization?

- Helps focus learning on **relative preference signals**, rather than noisy reward scores.
- Avoids need for value model or full trajectories.
- Robust to scale shifts in reward model outputs.

______________________________________________________________________

## 🤖 Where GRPO Shines

- When you want to **rank multiple outputs for hard reasoning tasks**.
- When training on **multi-step reasoning questions**, where final answer is too sparse as signal.
- When comparing several candidate generations is easier than labeling absolute scores or winning pairs.

______________________________________________________________________

## 📚 Origin

GRPO was introduced in:

> **"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"** (2024)

______________________________________________________________________

## ✅ Summary

GRPO is:

- A clean generalization of PPO for grouped outputs.
- Practical for reasoning tasks where multiple candidates are easier to evaluate relatively.
- Easier to train than full-on value-based PPO.
