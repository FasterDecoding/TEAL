import torch

def interp(x, xp, fp):
    """Custom interpolation function for PyTorch tensors."""
    i = torch.searchsorted(xp, x)
    i = torch.clamp(i, 1, len(xp) - 1)
    
    xp_left = xp[i - 1]
    xp_right = xp[i]
    fp_left = fp[i - 1]
    fp_right = fp[i]
    
    t = (x - xp_left) / (xp_right - xp_left)
    return fp_left + t * (fp_right - fp_left)

class Distribution:
    def __init__(self, file_path, hidden_type):
        self.file_path = file_path
        self.hidden_type = hidden_type # h1 or h2
        
        histogram = torch.load(f"{self.file_path}/histograms.pt")

        self.bin_centers, self.counts = histogram[f"{self.hidden_type}_centers"], histogram[self.hidden_type]

        self.total_count = self.counts.sum()
        self.cumulative_counts = torch.cumsum(self.counts, dim=0)

    # kernel smoothing
    def pdf(self, x, bandwidth=None):
        if bandwidth is None:
            bandwidth =  1.06 * torch.std(self.bin_centers[1:-1]) * (self.total_count-2)**(-1/5)
        
        bin_centers = self.bin_centers.unsqueeze(1)
        
        if isinstance(x, float) or isinstance(x, int):
            x = torch.tensor([x])
        else:
            x = x.unsqueeze(0)
        
        kernel = torch.exp(-0.5 * ((x - bin_centers) / bandwidth)**2) / (bandwidth * torch.sqrt(torch.tensor(2 * torch.pi)))
        pdf = torch.sum(kernel * self.counts.unsqueeze(1), dim=0) / self.total_count
        
        return pdf
    
    def cdf(self, x):
        return interp(x, self.bin_centers, self.cumulative_counts / self.total_count)
    
    def icdf(self, q):
        # if q < 0.01 or q > 0.99:
        #     print(f"WARNING: All outliers clip to the most extreme bin")

        target_count = q * self.total_count
        idx = torch.searchsorted(self.cumulative_counts, target_count)
        
        if idx == 0:
            return self.bin_centers[0]
        elif idx == len(self.bin_centers):
            return self.bin_centers[-1]
        else:
            lower_count = self.cumulative_counts[idx - 1]
            upper_count = self.cumulative_counts[idx]
            lower_value = self.bin_centers[idx - 1]
            upper_value = self.bin_centers[idx]
            
            fraction = (target_count - lower_count) / (upper_count - lower_count)
            return lower_value + fraction * (upper_value - lower_value)

    def abs_icdf(self, q):
        if q < 0 or q > 1:
            raise ValueError("q must be between 0 and 1")

        # Create a new histogram for the absolute values
        abs_bin_centers = torch.abs(self.bin_centers)
        abs_counts = torch.zeros_like(self.counts)

        # Combine counts for positive and negative values
        for i, center in enumerate(self.bin_centers):
            abs_index = torch.argmin(torch.abs(abs_bin_centers - abs(center)))
            abs_counts[abs_index] += self.counts[i]

        # Sort the absolute bin centers and counts
        sorted_indices = torch.argsort(abs_bin_centers)
        sorted_abs_centers = abs_bin_centers[sorted_indices]
        sorted_abs_counts = abs_counts[sorted_indices]

        # Calculate the cumulative distribution
        total_count = sorted_abs_counts.sum()
        cumulative_counts = torch.cumsum(sorted_abs_counts, dim=0)
        
        # Find the index where the cumulative probability exceeds q
        target_count = q * total_count
        idx = torch.searchsorted(cumulative_counts, target_count)

        if idx == 0:
            return sorted_abs_centers[0]
        elif idx == len(sorted_abs_centers):
            return sorted_abs_centers[-1]
        else:
            lower_count = cumulative_counts[idx - 1]
            upper_count = cumulative_counts[idx]
            lower_value = sorted_abs_centers[idx - 1]
            upper_value = sorted_abs_centers[idx]
            
            fraction = (target_count - lower_count) / (upper_count - lower_count)
            return lower_value + fraction * (upper_value - lower_value)
