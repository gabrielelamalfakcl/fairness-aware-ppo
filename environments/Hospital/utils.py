import random

def build_arrival_schedule(n_patients: int, peaks: list[tuple[int, int]], sim_minutes: int, λ_peak: float = 12, λ_off:  float = 3):
    """
    Returns a list of integer arrival-times (minutes) for n_patients.
    peaks – list of (start_minute, end_minute) intervals that are 'peak'
    λ_peak / λ_off – average arrivals per hour in / out of the peak
    """
    
    def is_peak(t):
        return any(a <= t < b for a, b in peaks)

    times, t = [], 0
    while len(times) < n_patients:
        lam  = λ_peak if is_peak(t) else λ_off
        gap  = random.expovariate(lam / 60) # convert to per-minute rate
        t    = int(t + gap)
        times.append(t)
        
    # We need to be sure the patients all arrive within the final minute of the simulation 
    last_arrival_time = times[-1]
    if last_arrival_time > sim_minutes:
        scale_factor = sim_minutes / last_arrival_time
        # Apply the scaling factor to every arrival time.
        # Subtract a small epsilon to ensure the last arrival is just before the limit.
        times = [int(time * scale_factor - 0.001) for time in times]
    else:
        # If the schedule already fits, just convert times to integers.
        times = [int(time) for time in times]
    return times
  
def generate_arrival_templates(n_templates, sim_minutes=480, seed=42):
    """
    Generates a list of unique daily arrival patterns with reduced variance.
    """
    
    random.seed(seed)
    templates = []
    
    # Define the parameters for a "typical" day
    mean_morning_peak_start = 120  # 10 AM
    mean_afternoon_peak_start = 300 # 3 PM
    peak_start_std_dev = 30 # Standard deviation of 30 minutes
    
    mean_peak_duration = 60 # Average peak is 1 hour
    peak_duration_std_dev = 15 # Standard deviation of 15 minutes

    for _ in range(n_templates):
        # Generate 2 peaks for consistency
        peaks = []
        
        # Morning Peak
        start1 = int(random.normalvariate(mean_morning_peak_start, peak_start_std_dev))
        duration1 = int(random.normalvariate(mean_peak_duration, peak_duration_std_dev))
        # Ensure values are within reasonable bounds
        start1 = max(0, min(start1, sim_minutes - 30))
        duration1 = max(30, min(duration1, 120))
        end1 = min(sim_minutes, start1 + duration1)
        peaks.append((start1, end1))

        # Afternoon Peak
        start2 = int(random.normalvariate(mean_afternoon_peak_start, peak_start_std_dev))
        duration2 = int(random.normalvariate(mean_peak_duration, peak_duration_std_dev))
        # Ensure values are within reasonable bounds
        start2 = max(0, min(start2, sim_minutes - 30))
        duration2 = max(30, min(duration2, 120))
        end2 = min(sim_minutes, start2 + duration2)
        peaks.append((start2, end2))
        
        peaks.sort()
        templates.append({"peaks": peaks})
        
    return templates