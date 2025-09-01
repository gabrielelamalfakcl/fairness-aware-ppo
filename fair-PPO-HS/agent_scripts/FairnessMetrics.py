import torch

class CumulativeMetrics:
    """
    Computes fairness metrics based on cumulative rewards.
    """

    def demographic_parity(self, total_rewards_non_sensitive, total_rewards_sensitive):
        """
        Computes demographic parity based on total cumulative rewards.
        """
        demographic_parity = total_rewards_non_sensitive - total_rewards_sensitive
        total_rewards = max(total_rewards_non_sensitive + total_rewards_sensitive, 1.0)  # Avoid division by zero
        norm_demographic_parity = demographic_parity / total_rewards
        
        return demographic_parity, norm_demographic_parity

    def conditional_statistical_parity(self, total_rewards_non_sensitive_G1, total_rewards_sensitive_G1,
        total_rewards_non_sensitive_G2, total_rewards_sensitive_G2):
        """
        Computes conditional statistical parity based on cumulative rewards.
        """
        csp_G1 = total_rewards_non_sensitive_G1 - total_rewards_sensitive_G1
        csp_G2 = total_rewards_non_sensitive_G2 - total_rewards_sensitive_G2

        total_rewards_G1 = max(total_rewards_non_sensitive_G1 + total_rewards_sensitive_G1, 1.0)
        total_rewards_G2 = max(total_rewards_non_sensitive_G2 + total_rewards_sensitive_G2, 1.0)

        norm_csp_G1 = csp_G1 / total_rewards_G1
        norm_csp_G2 = csp_G2 / total_rewards_G2
        
        return csp_G1, csp_G2, norm_csp_G1, norm_csp_G2

class StateValueMetrics:
    """
    Computes fairness metrics based on expected state values.
    """

    def demographic_parity(self, state_values_non_sensitive, state_values_sensitive):
        """
        Computes demographic parity based on average state values.
        """
        mean_not_protected = torch.mean(state_values_non_sensitive)
        mean_protected = torch.mean(state_values_sensitive)

        demographic_parity = mean_not_protected - mean_protected
        total_values = max(mean_not_protected + mean_protected, 1.0)  # Avoid division by zero
        norm_demographic_parity = demographic_parity / total_values
        
        return demographic_parity, norm_demographic_parity

    def conditional_statistical_parity(self, 
        state_values_non_sensitive_G1, state_values_sensitive_G1,
        state_values_non_sensitive_G2, state_values_sensitive_G2
    ):
        """
        Computes conditional statistical parity based on state values.
        """

        def ensure_tensor(value):
            return torch.tensor(value, dtype=torch.float32) if not isinstance(value, torch.Tensor) else value

        state_values_non_sensitive_G1 = ensure_tensor(state_values_non_sensitive_G1)
        state_values_sensitive_G1 = ensure_tensor(state_values_sensitive_G1)
        state_values_non_sensitive_G2 = ensure_tensor(state_values_non_sensitive_G2)
        state_values_sensitive_G2 = ensure_tensor(state_values_sensitive_G2)

        # Compute means
        mean_not_protected_G1 = torch.mean(state_values_non_sensitive_G1)
        mean_protected_G1 = torch.mean(state_values_sensitive_G1)
        mean_not_protected_G2 = torch.mean(state_values_non_sensitive_G2)
        mean_protected_G2 = torch.mean(state_values_sensitive_G2)

        # Compute CSP values
        csp_G1 = mean_not_protected_G1 - mean_protected_G1
        csp_G2 = mean_not_protected_G2 - mean_protected_G2

        # Ensure denominator is not zero
        total_values_G1 = max(mean_not_protected_G1 + mean_protected_G1, torch.tensor(1.0, dtype=torch.float32))
        total_values_G2 = max(mean_not_protected_G2 + mean_protected_G2, torch.tensor(1.0, dtype=torch.float32))

        # Compute normalized CSP
        norm_csp_G1 = csp_G1 / total_values_G1
        norm_csp_G2 = csp_G2 / total_values_G2
        
        return csp_G1, csp_G2, norm_csp_G1, norm_csp_G2