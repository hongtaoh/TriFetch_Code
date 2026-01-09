import torch
import torch.nn.functional as F

def compute_grpo_advantage(rewards, epsilon=1e-6):
    """
    Computes Group Relative Policy Optimization (GRPO) Advantages.
    
    Instead of using a Value Network (Critic) to estimate the baseline,
    GRPO uses the average reward of the 'group' of sampled traces as the baseline.
    
    Formula:
        A_i = (r_i - mean(R)) / (std(R) + epsilon)
    
    Args:
        rewards: Raw rewards for the group (e.g., [1.0, 0.5, 0.0])
        epsilon: Small constant to prevent division by zero.
    
    Returns:
        torch.Tensor: Normalized advantages for each trace.
    """
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32)
    
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    
    advantages = (rewards - mean_reward) / (std_reward + epsilon)
    
    return advantages


def compute_dpo_loss(policy_chosen_logps, policy_rejected_logps,
                     ref_chosen_logps, ref_rejected_logps,
                     beta=0.1):
    """
    Computes Direct Preference Optimization (DPO) Loss.
        
    DPO optimizes the policy by increasing the likelihood of 'chosen' responses
    relative to 'rejected' responses, anchored by a reference model.
    
    Formula:
        Loss = -log(sigmoid(beta * ((log_π(chosen) - log_π(rejected)) - 
                                     (log_π_ref(chosen) - log_π_ref(rejected)))))
    
    Args:
        policy_chosen_logps: Log-prob of policy model on winning trace
        policy_rejected_logps: Log-prob of policy model on losing trace
        ref_chosen_logps: Log-prob of reference model on winning trace
        ref_rejected_logps: Log-prob of reference model on losing trace
        beta: Temperature parameter (typically 0.1 to 0.5)
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    # Ensure tensors
    if not isinstance(policy_chosen_logps, torch.Tensor):
        policy_chosen_logps = torch.tensor(policy_chosen_logps, dtype=torch.float32)
    if not isinstance(policy_rejected_logps, torch.Tensor):
        policy_rejected_logps = torch.tensor(policy_rejected_logps, dtype=torch.float32)
    if not isinstance(ref_chosen_logps, torch.Tensor):
        ref_chosen_logps = torch.tensor(ref_chosen_logps, dtype=torch.float32)
    if not isinstance(ref_rejected_logps, torch.Tensor):
        ref_rejected_logps = torch.tensor(ref_rejected_logps, dtype=torch.float32)
    
    # Policy model's preference margin
    policy_log_ratios = policy_chosen_logps - policy_rejected_logps
    
    # Reference model's preference margin (baseline)
    ref_log_ratios = ref_chosen_logps - ref_rejected_logps
    
    # DPO logits: how much more does policy prefer chosen vs reference
    logits = policy_log_ratios - ref_log_ratios
    
    # Negative log-sigmoid loss
    losses = -F.logsigmoid(beta * logits)
    
    return losses.mean()