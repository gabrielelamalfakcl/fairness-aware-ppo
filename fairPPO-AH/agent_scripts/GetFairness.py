class GetMetrics:
    def __init__(self):
        None
        
    def demographic_parity(self, total_rewards_not_protected, total_rewards_protected):
        """
        Compute demographic parity metric on average rewards for each episode.
        Normalize rewards to bring them on a comparable scale to state values.
        """
        normalized_not_protected = total_rewards_not_protected
        normalized_protected = total_rewards_protected
        
        demographic_parity = (normalized_not_protected - normalized_protected)
        total_rewards = max(normalized_not_protected + normalized_protected, 1)
        norm_demographic_parity = demographic_parity / total_rewards
        
        return demographic_parity, norm_demographic_parity
    
    def conditional_statistical_parity(self, total_rewards_not_protected_G1, total_rewards_protected_G1, 
                                    total_rewards_not_protected_G2, total_rewards_protected_G2):
        """
        Compute conditional stat parity metric on average rewards for each episode.
        Normalize rewards to bring them on a comparable scale to state values.
        Input: cumulative rewards for each episode for the two groups.
        Compute the cumulative rewards for each subgroup (indicated by val_attribute).
        Output: conditional stat parity metric (single value for each subgroup for each episode).
        """
        
        # Calculate conditional statistical parity for G1 and G2
        conditional_stat_parity_G1 = (total_rewards_not_protected_G1 - total_rewards_protected_G1)
        conditional_stat_parity_G2 = (total_rewards_not_protected_G2 - total_rewards_protected_G2)
        
        # Total rewards for normalization
        total_rewards_G1 = max(total_rewards_not_protected_G1 + total_rewards_protected_G1, 1)
        total_rewards_G2 = max(total_rewards_not_protected_G2 + total_rewards_protected_G2, 1)
        
        # Normalized conditional statistical parity
        norm_csp_G1 = conditional_stat_parity_G1 / total_rewards_G1
        norm_csp_G2 = conditional_stat_parity_G2 / total_rewards_G2
        
        return conditional_stat_parity_G1, conditional_stat_parity_G2, norm_csp_G1, norm_csp_G2


        
        
        


