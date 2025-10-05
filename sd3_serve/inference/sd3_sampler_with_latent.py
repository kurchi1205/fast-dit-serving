import torch
import torch.nn as nn

class ModelSamplingDiscreteFlowAdaptive(nn.Module):
    """Discrete Flow Sampler with optional latent-based adaptive sigma scheduling"""
    
    def __init__(self, shift=1.0):
        super().__init__()
        self.shift = shift
        timesteps = 1000
        ts = self.sigma(torch.arange(1, timesteps + 1, 1))
        self.register_buffer("sigmas", ts)

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def get_schedule(self, num_steps):
        start = self.timestep(self.sigmas[0])
        end = self.timestep(self.sigmas[-1])
        timesteps = torch.linspace(start, end, num_steps)
        sigs = []
        for x in range(len(timesteps)):
            ts = timesteps[x]
            sigs.append(self.sigma(ts))
        return torch.FloatTensor(sigs)

    def timestep(self, sigma):
        return sigma * 1000

    def sigma(self, timestep: torch.Tensor):
        # Standard schedule based on shift
        timestep = timestep / 1000.0
        if self.shift == 1.0:
            return timestep
        return (self.shift * (timestep ** 1.2)) / (1 + (self.shift - 1) * (timestep ** 1.2))

    def sigma_from_latent(self, latent: torch.Tensor, base_sigma: float, min_factor=0.7, max_factor=1):
        """
        Simplified version with just the most effective metrics.
        """
        # Calculate metrics
        std_complexity = latent.std(dim=(1, 2, 3), keepdim=False)
        
        grad_x = torch.diff(latent, dim=-1)
        grad_y = torch.diff(latent, dim=-2)
        min_h = min(grad_x.shape[-2], grad_y.shape[-2])
        min_w = min(grad_x.shape[-1], grad_y.shape[-1])
        grad_x_cropped = grad_x[..., :min_h, :min_w]
        grad_y_cropped = grad_y[..., :min_h, :min_w]
        grad_magnitude = torch.sqrt(grad_x_cropped.pow(2) + grad_y_cropped.pow(2))
        edge_complexity = grad_magnitude.mean(dim=(1, 2, 3))
        
        edge_too_low = (edge_complexity < 0.9) & (edge_complexity > 0.7) # [B]
        if edge_too_low.any():
            noise = torch.randn_like(latent) * 0.01
            latent[edge_too_low] = latent[edge_too_low] + noise[edge_too_low]
            del noise
        
        # edge_norm = torch.clamp(edge_complexity, min=0.4) / 0.5  # Floor edge at 0.4
        edge_norm = edge_complexity/ 0.5  # Floor edge at 0.4
        
        # Your formula
        adjustment = std_complexity + 0.3 * edge_complexity
        
        # Your edge penalty (using original edge_complexity)
        edge_penalty = torch.sigmoid((edge_complexity) * 2.0)
        boost = 1.0 + edge_penalty * 0.3
        adjustment = adjustment * boost
        
        # Final clamp
        adjustment = torch.sigmoid((adjustment))
        adjustment = min_factor + (max_factor - min_factor) * adjustment
        return base_sigma * adjustment, latent

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma

    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        """
        Combines latent and noise based on sigma.
        Now allows sigma to be a tensor for adaptive per-sample noise.
        """
        return sigma * noise + (1.0 - sigma) * latent_image
