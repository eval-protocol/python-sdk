# GiGPO Reward Function Pseudocode Breakdown

GiGPO (Group-in-Group Policy Optimization) introduces a two-level reward/advantage structure for policy optimization. Below we break down the reward function logic step by step, focusing on how episode-level and step-level relative advantages are computed, normalized, and integrated into the policy loss. We provide conceptual pseudocode with comments, marking **(Confirmed)** for components directly from official sources and **(Inferred)** for logical extrapolations.

## Episode-Level Relative Advantage (Macro Advantage)

GiGPO first forms **episode-level groups** of trajectories that share the same initial state. All trajectories in a group start from an identical environment state and represent attempts of the agent to complete the same task. We measure each trajectory’s overall success by its total return $R_i$ (the sum of rewards over the episode). For example, in a sparse-reward task with a binary outcome, $R_i = 1$ for success and $0$ for failure.

Within each group of trajectories, GiGPO computes a **relative advantage** for each episode by comparing its return to the group’s statistics. The intuition (as in GRPO) is to reward trajectories that outperformed their peers and penalize those that underperformed. Concretely, the episode advantage $A^E_i$ for trajectory $i$ is the trajectory’s return, normalized by the group’s mean return and a normalization factor:

* By default, the normalization factor is the group’s standard deviation (as in GRPO), which yields a **z-score** style advantage. This makes $A^E_i = \frac{R_i - \text{mean}(R_{\text{group}})}{\text{std}(R_{\text{group}})}$ **(Confirmed)**.
* GiGPO also considers using a fixed constant $F$ instead of the std, to avoid extreme advantage magnitudes when the group’s return variance is very low (a phenomenon called *difficulty bias* in group-based RL). Using a constant normalization (e.g. a fixed scalar or one derived from group size) yields an *unbiased leave-one-out* advantage estimator. We denote this option in pseudocode for completeness **(Confirmed)**.

The pseudocode below illustrates computing episode-level advantages for a group of trajectories. Each trajectory’s episode advantage acts as a **macro-level reward signal** indicating how well that entire trajectory did relative to others.

```python
# Assume we have N trajectories in a group, all with identical initial state (Confirmed:contentReference[oaicite:15]{index=15})
trajectories = [traj_1, traj_2, ..., traj_N]

# Compute total return for each trajectory (sum of rewards over the episode)
returns = [sum(traj.rewards) for traj in trajectories]  # (Confirmed: total return per traj:contentReference[oaicite:16]{index=16})

# Calculate group statistics
mean_return = mean(returns)
if normalization_mode == "std":
    norm_factor = stdev(returns)  # (Confirmed: default normalization is std:contentReference[oaicite:17]{index=17})
else:
    norm_factor = const_F        # (Confirmed: fixed factor to avoid bias:contentReference[oaicite:18]{index=18}, exact value from RLOO)

# Compute episode-level advantages for each trajectory
episode_advantages = []
for R_i in returns:
    # Episode advantage A^E_i = (R_i - mean_return) / norm_factor
    A_E_i = (R_i - mean_return) / norm_factor
    episode_advantages.append(A_E_i)
    # ^ Confirmed from source: advantage normalized by group's mean & factor:contentReference[oaicite:19]{index=19}
```

**Notes:** All trajectories in the group share the same initial state **(Confirmed)**, ensuring the comparison is fair. If the task provides only a final success/fail reward, most returns will be 0 or 1; the normalization then highlights even small differences across a batch of trials. The result is an **episode advantage vector** $A^E = \{A^E_1, ..., A^E_N\}$ giving each trajectory a relative score. This episode-level signal captures **global task completion quality** – it will encourage policies that achieve higher returns (e.g. successful outcomes) over those that don’t.

## Step-Level Relative Advantage (Micro Advantage)

Episode-level advantages alone treat each trajectory as a whole and don’t tell us which **specific actions** were good or bad within the episode. GiGPO introduces a **step-level grouping** mechanism to assign fine-grained credit to individual actions. The key idea is to compare actions taken in the **same state** across different trajectories, so we can evaluate which action choices led to better outcomes from that state.

**Anchor state grouping:** Since all trajectories start from the same conditions, many environment states recur across different episodes (and even multiple times in one episode). GiGPO leverages this by using each unique state encountered as an *anchor* for grouping. For each distinct state $s$ that appears in the trajectory set, we gather **all occurrences** of $s$ along with the action taken in each occurrence. This produces a *state-centric group* $G_S(s)$ = { (action, outcome) pairs from every time state \$s\$ was seen }. Essentially, we collate all instances where the agent was in state \$s\$ and record what action was taken and what result followed.

**Measuring outcomes for actions:** If the environment provides immediate step rewards, those alone might be sparse or delayed signals of an action’s quality. GiGPO instead computes a **discounted return from that step onward** to estimate the long-term outcome of each action taken at an anchor state. For an occurrence of state \$s\$ at time \$t\$ in trajectory \$i\$, with action \$a\_t^{(i)}\$, we compute:

$R^{(i)}_t = \sum_{k=t}^{T} \gamma^{\,k-t} \, r^{(i)}_k,$

the cumulative future reward from time \$t\$ (including the immediate reward \$r^{(i)}\_t\$) discounted by factor \$\gamma\$. This \$R^{(i)}\_t\$ represents how well things turned out *after taking that action* at state \$s\$ (higher if that action eventually led to success, lower if it led to failure or low reward). We then form the **step-level group** for state \$s\$ as $G_S(s) = \{(a_t^{(i)}, R^{(i)}_t) \mid s_t^{(i)} = s\}$ – all recorded actions from \$s\$ paired with their subsequent returns.

Now we compute the **step relative advantage** for each action occurrence in these state-based groups. This parallels the episode advantage computation, but at the level of a specific state:

```python
# Build anchor-state groups for step-level comparisons (Confirmed logic:contentReference[oaicite:38]{index=38}):
anchor_groups = {}  # maps state -> list of (action, discounted_return)
for i, traj in enumerate(trajectories):
    for t, (s, a, r) in enumerate(traj.steps):
        # Each step has state s, action a, immediate reward r
        if s not in anchor_groups:
            anchor_groups[s] = []
        # Compute discounted return from this step (Confirmed formula:contentReference[oaicite:39]{index=39})
        future_return = 0
        discount = 1
        for k in range(t, len(traj.steps)):  # sum rewards from t to end
            r_k = traj.steps[k].reward
            future_return += discount * r_k
            discount *= gamma
        # Store the action and its outcome in the state group
        anchor_groups[s].append((a, future_return))
        # ^ Confirmed: group all actions taken from the same state with their returns:contentReference[oaicite:40]{index=40}:contentReference[oaicite:41]{index=41}

# Compute step-level relative advantages within each anchor state group:
step_advantages = {}  # maps state -> list of advantages for each action occurrence in that state
for s, occurrences in anchor_groups.items():
    # Each occurrence is (action, discounted_return)
    returns_s = [R for (_, R) in occurrences]
    mean_R = mean(returns_s)
    if normalization_mode == "std":
        norm_factor = stdev(returns_s)
    else:
        norm_factor = const_F  # (Inferred: could reuse same fixed factor idea for consistency)
    # Compute advantage for each action taken from state s
    adv_list = []
    for (a, R_val) in occurrences:
        A_S = (R_val - mean_R) / norm_factor
        adv_list.append(A_S)
        # ^ Inferred: normalize like episode-level (mean & std) to get relative advantage of this action
    step_advantages[s] = adv_list
```

In the pseudocode above, we use a dictionary to group states (this reflects the described “hashmap” grouping in the paper **(Confirmed)**). For each state \$s\$ encountered, we collect all \$(a, R\_{\text{future}})\$ pairs. Then for each state-group, we compute the average outcome and normalize each action’s outcome to get its **step advantage** \$A^S(s, a)\$ as a z-score within that state’s group (i.e. relative to other actions taken from the same state). This gives a **micro-level feedback signal**: if a particular action led to a much higher future return compared to alternative actions from the same state, it gets a high positive \$A^S\$; if it fared worse than others, it gets a negative \$A^S\$. (In practice, the GiGPO paper implies the same normalization approach as episode-level is used here, which we have assumed in pseudocode **(Inferred)**.)

**Example:** Figure 3 of the GiGPO paper illustrates this with a web navigation task. Two trajectories both reach a search-results page (the anchor state). One trajectory clicks the **“2nd Item”** then later the **“1st Item”** (which leads to success), and another trajectory clicks **“Next Page”** (leading to failure). By grouping these actions at the same page state, GiGPO can assign relative advantages: the successful **1st Item** click receives a higher discounted return (less discount and leads to success) than the **2nd Item**, which in turn is better than **Next Page** (which led to no reward). The computed \$A^S\$ would rank these actions accordingly as $A^S(\text{1st Item}) > A^S(\text{2nd Item}) > A^S(\text{Next Page})$. This fine-grained ranking is something vanilla episode-level methods would miss.

## Combining Macro and Micro Advantages

After computing both the episode-level advantage \$A^E\_i\$ for each trajectory and the step-level advantages \$A^S\$ for each action, GiGPO **blends these two signals** into a single **group-in-group advantage** for each state-action occurrence. This combined advantage provides **hierarchical supervision**, capturing both the trajectory’s overall quality and the action’s relative merit in its state.

We introduce a weighting coefficient (call it \$\omega\$) to balance the two levels. There are a few ways to combine them; conceptually GiGPO uses a linear combination:

$A^{GiG}(s, a) \;=\; \omega \cdot A^E(\text{trajectory}(s,a)) \;+\; (1-\omega) \cdot A^S(s, a) ,$

where \$A^E(\text{trajectory}(s,a))\$ is the episode advantage of the trajectory in which action \$a\$ was taken, and \$A^S(s,a)\$ is the step advantage for that specific state-action. This formula reflects giving a fraction of credit to the trajectory’s overall success and a fraction to the action’s state-specific performance. (In the paper, \$\omega\$ is described as a non-negative balancing coefficient; if \$\omega=0\$, we rely purely on step-level credit, and if \$\omega\$ is large or 1, more weight is on the global outcome.)

**Usage in practice:** In each trajectory, for each step $t$ where the agent took action $a_t$ in state $s_t$, we can now assign a final advantage signal $A^{GiG}_{t}$. The episode part $A^E_i$ is the *same for all steps of that trajectory* (a constant macro-level feedback for that episode) **(Inferred)**, and \$A^S\_{t}\$ is the tailored adjustment for that particular action at state \$s\_t\$. The pseudocode below shows how we might compute the combined advantage for each action in the trajectories:

```python
# Assume we have episode_advantages list and step_advantages dict from earlier.
omega = 0.5  # example weight (could be tuned)

# For each trajectory, assign combined advantage to each action step
combined_advantages = []  # list of (traj_index, step_index, combined_adv) for all actions
for i, traj in enumerate(trajectories):
    A_E = episode_advantages[i]      # Episode-level advantage for trajectory i (Confirmed:contentReference[oaicite:54]{index=54})
    for t, step in enumerate(traj.steps):
        state = step.state
        # Find the corresponding step advantage for this occurrence.
        # We must retrieve the t-th occurrence in the anchor group list for state.
        # (For simplicity, assume anchor_groups[s] order matches trajectory order; otherwise we'd match by action id.)
        A_S = step_advantages[state].pop(0)  # take the next advantage value for state (Inferred retrieval)
        # Compute weighted combined advantage
        A_comb = omega * A_E + (1 - omega) * A_S
        combined_advantages.append((i, t, A_comb))
        # ^ Confirmed combination logic: blends episode (global) and step (local) advantages:contentReference[oaicite:55]{index=55}
```

In this pseudocode, we iterate through each trajectory and each step, then look up the precomputed \$A^S\$ for that state-action. We combine it with the trajectory’s \$A^E\$. (The code uses a simple method to retrieve matching \$A^S\$ values; in an actual implementation, we’d ensure we get the correct advantage corresponding to that exact state occurrence, e.g. by storing indices or iterating similarly as when we built the groups.) The result $A^{GiG}$ serves as the **final advantage (reward signal) for that specific action** in training.

Importantly, \$A^E\$ injects a **global success signal** into every step of a successful trajectory (so all actions in a winning episode get nudged up, and all actions in a failing episode get nudged down), while \$A^S\$ fine-tunes the feedback for each action relative to alternatives at that state. The weighting \$\omega\$ can be adjusted: a higher \$\omega\$ emphasizes overall success/failure more strongly, and a lower \$\omega\$ focuses more on per-step decisions.

## Policy Optimization with GiGPO Advantage

Finally, GiGPO uses the combined advantage values in a policy optimization objective. GiGPO’s training objective is based on the PPO clipped surrogate loss, but replacing the usual value-function-based advantage with our **group-in-group advantage** \$A^{GiG}\$. In other words, it is a **critic-free** update that leverages these computed advantages directly.

At a high level, for each action taken in our collected trajectories, we have an advantage \$A^{GiG}*{t}\$. We also have the probability ratio \$\rho\_t = \frac{\pi*\theta(a\_t|s\_t)}{\pi\_{\text{old}}(a\_t|s\_t)}\$ between the current policy and the policy that generated the trajectory (or a reference policy). GiGPO maximizes the expected advantage-weighted log probability of actions, using PPO-style clipping to ensure stable updates. Conceptually, the loss (to minimize) for a single action can be written as:

$ L_{\text{clip}}(a_t) = -\min\!\Big(\rho_t \cdot A^{GiG}_{t},\;\; \text{clip}(\rho_t,\,1-\epsilon,\,1+\epsilon)\cdot A^{GiG}_{t}\Big) ,$

where \$\epsilon\$ is the PPO clip range. This is identical to PPO’s surrogate loss, except using our custom advantage numbers. The overall objective sums this over all time steps and trajectories in the batch. In pseudocode form:

```python
# Policy update phase: use combined advantages in a PPO-style loss
loss = 0
for (i, t, A_comb) in combined_advantages:
    state = trajectories[i].steps[t].state
    action = trajectories[i].steps[t].action
    # Compute probability ratio (importance sampling) for this action under the new policy vs old policy
    pi_new = policy.prob(action, state)
    pi_old = trajectories[i].steps[t].old_prob  # probability under the old policy (stored from rollout)
    rho = pi_new / pi_old
    # PPO clipped surrogate term for this action (Confirmed formulation:contentReference[oaicite:62]{index=62})
    unclipped = rho * A_comb
    clipped = clip(rho, 1 - epsilon, 1 + epsilon) * A_comb
    # We take negative because we typically minimize the loss (gradient ascent on reward)
    loss += -min(unclipped, clipped)
# (Optionally, add a KL penalty to keep policy close to a reference model, if applicable)
loss += beta * KL_divergence(policy, reference_policy)  # (Confirmed: KL regularization term:contentReference[oaicite:63]{index=63})

# Backpropagate loss and update policy parameters...
```

In this snippet, each action’s contribution to the loss is scaled by its combined advantage \$A^{GiG}\$ and clipped per PPO’s rule. The term with `KL_divergence` is included because in LLM agent training, it’s common to penalize the policy for straying too far from an initial policy (to maintain fluent language, etc.). The GiGPO paper indeed includes a KL penalty with coefficient \$\beta\$ in its objective. We mark that as confirmed, since they specifically mention a term encouraging the policy to stay close to a reference policy (often the pre-trained model before RL fine-tuning).

**Summary:** GiGPO’s reward function logic provides two levels of advantage signals – **episode-level (macro)** and **step-level (micro)** – which are normalized and combined to guide policy updates. The episode advantage \$A^E\$ tells the agent which entire trajectories were successful relative to peers, and the step advantage \$A^S\$ tells which action choices were better or worse when encountering the same situation. By integrating both, the final advantage \$A^{GiG}\$ provides a richer reward signal for each action, helping credit assignment in long-horizon tasks without a critic network. This combined advantage is then plugged into a PPO-style optimization, ensuring stable training with clipping and regularization. The result is a policy gradient update that **reinforces trajectories that succeeded** and **actions that proved optimal** in their contexts, all computed from **group-wise comparisons** of the agent’s own rollouts instead of a learned value function.

**Sources:** The above pseudocode and explanation are based on the GiGPO algorithm as described by the authors, with standard PPO practices for the loss function. All core computations (grouping by identical initial state, relative advantage = normalized return, anchor state grouping, discounted returns for step outcomes, and advantage combination) are directly drawn from the official paper and code descriptions. Minor implementation details (like data structures for grouping and applying the same episode advantage to each step) are inferred for clarity, aligning with typical usage of these signals in policy optimization. The combined approach allows GiGPO to deliver fine-grained per-step feedback and improved performance over baseline GRPO, without additional environment rollouts or a value critic.

-----

Follow up question from me:

Wait, does that mean the trainer somehow need to support different reward for different tokens in the same rollout??

-----

Excellent observation — and yes, **GiGPO implicitly requires that the trainer support *per-token (or per-step)* rewards within the same rollout**.

Here’s how that works and what’s actually required:

---

### **1. Standard PPO Assumption**

In vanilla PPO (especially in RLHF for LLMs), you often:

* Treat the **entire generated sequence** as *one action*,
* Assign a **single scalar reward** to the whole thing,
* And optimize only the part the model generated (ignoring prompt or tool outputs).

That’s fine for **single-turn** rollouts or very short sequences.

---

### **2. GiGPO’s Requirement**

In GiGPO, a rollout is a **sequence of multiple steps**, each of which includes:

* A state (e.g. conversation history, environment state),
* An action (e.g. token span or tool call),
* A computed **advantage specific to that step**: `A_GiG[t]`.

So yes — to use GiGPO:

* The **trainer must apply a different advantage (reward signal)** at **each time step (or token span)** in the trajectory.
* The loss function is applied per step:

  ```python
  loss = sum_t -min(rho[t] * A_GiG[t], clipped_rho[t] * A_GiG[t])
  ```

**This is fundamentally a per-step loss**, not a single-shot reward over a whole rollout.

---

### **3. Implications for LLMs**

If your rollout is made of:

* **Text steps** (like `prompt → tool call → tool output → next tool call → ...`),
* Then you might represent **each step as a separate action chunk** (not necessarily individual tokens),
* And apply its own advantage to that chunk.

You don’t need to assign a reward to every *token* if your steps are at a higher granularity (like turns or tool calls).

---

### **4. Implementation Notes**

To implement this:

* Your `Rollout` data structure should store a list of `(state_t, action_t, advantage_t, logprob_t)` tuples,
* Your training loop applies the PPO loss **across this list**,
* Each action gets its **own weight (advantage)** based on GiGPO’s step+episode blending.

So:

* **Yes**, the trainer must support different advantages per step.
* But **no**, you don’t necessarily need to reward each token — just each decision/action at the RL level.

---

### TL;DR

**Yes**, GiGPO requires **per-step reward signals** within a trajectory.
**No**, you don’t have to assign one reward per token — just per *logical action step* (e.g., tool call, reasoning turn).
This means your trainer must support computing and applying **advantages step-by-step** rather than one reward per rollout.
