import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from pathlib import Path
import json

from muse_maskgit_pytorch import VQGanVAE, MaskGit, MaskGitTransformer
from muse_maskgit_pytorch.dna_encoder import OneHotDNAEncoder
from muse_maskgit_pytorch.muse_maskgit_pytorch import generate_from_dna

class HiCMapEvaluator:
    def __init__(self, model_path, vae_path, genome_fasta):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load models
        self._load_models(model_path, vae_path, genome_fasta)
        
        # Evaluation metrics storage
        self.results = {}
        
    def _load_models(self, model_path, vae_path, genome_fasta):
        """Load trained MaskGit and VAE models"""
        # DNA encoder
        self.dna_encoder = OneHotDNAEncoder(genome_fasta)
        
        # VAE
        self.vae = VQGanVAE(
            dim=256,
            codebook_size=1024,
            use_vgg_and_gan=False
        ).to(self.device)
        self.vae.load(vae_path)
        
        # Transformer
        self.transformer = MaskGitTransformer(
            num_tokens=1024,
            dim=768,  # Match your training config
            seq_len=1024,
            depth=12,
            heads=12,
            dim_head=64,
            dna_encoder=self.dna_encoder,
        ).to(self.device)
        
        # MaskGit
        self.maskgit = MaskGit(
            vae=self.vae,
            transformer=self.transformer,
            image_size=512,
            cond_image_size=256,
            cond_drop_prob=0.1,
        ).to(self.device)
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.maskgit.load_state_dict(checkpoint)
        self.maskgit.eval()
        
    def calculate_contact_distance_correlation(self, real_map, generated_map):
        """Calculate correlation between contact probability and genomic distance"""
        # Get upper triangle of the matrix
        n = real_map.shape[0]
        real_contacts = real_map[np.triu_indices(n, k=1)]
        gen_contacts = generated_map[np.triu_indices(n, k=1)]
        
        # Calculate genomic distances
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                distances.append(abs(i - j))
        
        # Calculate correlations
        real_corr, _ = pearsonr(distances, real_contacts)
        gen_corr, _ = pearsonr(distances, gen_contacts)
        
        return real_corr, gen_corr
    
    def calculate_structural_similarity(self, real_map, generated_map):
        """Calculate structural similarity between real and generated maps"""
        # Normalize maps
        real_norm = (real_map - real_map.mean()) / (real_map.std() + 1e-8)
        gen_norm = (generated_map - generated_map.mean()) / (generated_map.std() + 1e-8)
        
        # Calculate correlation
        correlation, _ = pearsonr(real_norm.flatten(), gen_norm.flatten())
        
        return correlation
    
    def calculate_topological_domains(self, hic_map, threshold_percentile=90):
        """Detect topological domains using insulation score"""
        n = hic_map.shape[0]
        insulation_scores = []
        
        for i in range(1, n-1):
            # Calculate local insulation score
            local_region = hic_map[max(0, i-10):min(n, i+11), 
                                  max(0, i-10):min(n, i+11)]
            insulation = np.mean(local_region)
            insulation_scores.append(insulation)
        
        # Find domains (low insulation regions)
        threshold = np.percentile(insulation_scores, threshold_percentile)
        domains = [i+1 for i, score in enumerate(insulation_scores) if score < threshold]
        
        return len(domains), domains
    
    def calculate_contact_decay(self, hic_map):
        """Calculate contact probability decay with distance"""
        n = hic_map.shape[0]
        distances = []
        contacts = []
        
        for d in range(1, n):
            diagonal_contacts = []
            for i in range(n - d):
                diagonal_contacts.append(hic_map[i, i + d])
            distances.append(d)
            contacts.append(np.mean(diagonal_contacts))
        
        return distances, contacts
    
    def evaluate_single_map(self, real_map, generated_map, map_id=""):
        """Evaluate a single Hi-C map pair"""
        results = {}
        
        # Basic statistics
        results['real_mean'] = np.mean(real_map)
        results['real_std'] = np.std(real_map)
        results['gen_mean'] = np.mean(generated_map)
        results['gen_std'] = np.std(generated_map)
        
        # Correlation metrics
        results['pearson_corr'] = pearsonr(real_map.flatten(), generated_map.flatten())[0]
        results['spearman_corr'] = spearmanr(real_map.flatten(), generated_map.flatten())[0]
        
        # Error metrics
        results['mse'] = mean_squared_error(real_map.flatten(), generated_map.flatten())
        results['mae'] = mean_absolute_error(real_map.flatten(), generated_map.flatten())
        
        # Structural similarity
        results['structural_sim'] = self.calculate_structural_similarity(real_map, generated_map)
        
        # Contact distance correlation
        real_dist_corr, gen_dist_corr = self.calculate_contact_distance_correlation(real_map, generated_map)
        results['real_dist_corr'] = real_dist_corr
        results['gen_dist_corr'] = gen_dist_corr
        
        # Topological domains
        real_domains, _ = self.calculate_topological_domains(real_map)
        gen_domains, _ = self.calculate_topological_domains(generated_map)
        results['real_domains'] = real_domains
        results['gen_domains'] = gen_domains
        results['domain_diff'] = abs(real_domains - gen_domains)
        
        # Contact decay
        real_distances, real_contacts = self.calculate_contact_decay(real_map)
        gen_distances, gen_contacts = self.calculate_contact_decay(generated_map)
        results['contact_decay_corr'] = pearsonr(real_contacts, gen_contacts)[0]
        
        return results
    
    def generate_and_evaluate(self, test_coords, test_images_lowres, test_images_highres):
        """Generate Hi-C maps and evaluate them against ground truth"""
        print("Generating Hi-C maps...")
        
        all_results = []
        
        for i, (coord, lowres, highres) in enumerate(zip(test_coords, test_images_lowres, test_images_highres)):
            print(f"Processing sample {i+1}/{len(test_coords)}")
            
            # Generate high-resolution map
            with torch.no_grad():
                generated = generate_from_dna(
                    self.maskgit,
                    dna_coords=[coord],
                    cond_images=lowres.unsqueeze(0).to(self.device),
                    cond_scale=3.0,
                    temperature=1.0,
                    timesteps=18
                )
                
                generated_map = generated[0, 0].cpu().numpy()  # Remove batch and channel dims
                real_map = highres[0].numpy()  # Remove channel dim
            
            # Evaluate
            results = self.evaluate_single_map(real_map, generated_map, f"sample_{i}")
            results['sample_id'] = i
            results['coord'] = coord
            all_results.append(results)
            
            # Save sample visualizations
            if i < 5:  # Save first 5 samples
                self._save_sample_visualization(real_map, generated_map, i, coord)
        
        # Aggregate results
        self.results = self._aggregate_results(all_results)
        
        return self.results
    
    def _save_sample_visualization(self, real_map, generated_map, sample_id, coord):
        """Save visualization of real vs generated maps"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Real map
        im1 = axes[0].imshow(real_map, cmap='Reds', vmax=np.percentile(real_map, 95))
        axes[0].set_title(f'Real Hi-C Map\n{coord}')
        plt.colorbar(im1, ax=axes[0])
        
        # Generated map
        im2 = axes[1].imshow(generated_map, cmap='Reds', vmax=np.percentile(generated_map, 95))
        axes[1].set_title(f'Generated Hi-C Map\n{coord}')
        plt.colorbar(im2, ax=axes[1])
        
        # Difference
        diff = np.abs(real_map - generated_map)
        im3 = axes[2].imshow(diff, cmap='viridis')
        axes[2].set_title('Absolute Difference')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.savefig(f'sample_{sample_id}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _aggregate_results(self, all_results):
        """Aggregate evaluation results across all samples"""
        df = pd.DataFrame(all_results)
        
        aggregated = {
            'num_samples': len(all_results),
            'mean_pearson_corr': df['pearson_corr'].mean(),
            'std_pearson_corr': df['pearson_corr'].std(),
            'mean_spearman_corr': df['spearman_corr'].mean(),
            'std_spearman_corr': df['spearman_corr'].std(),
            'mean_mse': df['mse'].mean(),
            'std_mse': df['mse'].std(),
            'mean_mae': df['mae'].mean(),
            'std_mae': df['mae'].std(),
            'mean_structural_sim': df['structural_sim'].mean(),
            'std_structural_sim': df['structural_sim'].std(),
            'mean_contact_decay_corr': df['contact_decay_corr'].mean(),
            'std_contact_decay_corr': df['contact_decay_corr'].std(),
            'mean_domain_diff': df['domain_diff'].mean(),
            'std_domain_diff': df['domain_diff'].std(),
        }
        
        return aggregated
    
    def save_results(self, output_path):
        """Save evaluation results to file"""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*50)
        print("Hi-C Map Generation Evaluation Summary")
        print("="*50)
        print(f"Number of samples evaluated: {self.results['num_samples']}")
        print(f"\nCorrelation Metrics:")
        print(f"  Pearson correlation: {self.results['mean_pearson_corr']:.4f} ± {self.results['std_pearson_corr']:.4f}")
        print(f"  Spearman correlation: {self.results['mean_spearman_corr']:.4f} ± {self.results['std_spearman_corr']:.4f}")
        print(f"  Structural similarity: {self.results['mean_structural_sim']:.4f} ± {self.results['std_structural_sim']:.4f}")
        
        print(f"\nError Metrics:")
        print(f"  Mean squared error: {self.results['mean_mse']:.4f} ± {self.results['std_mse']:.4f}")
        print(f"  Mean absolute error: {self.results['mean_mae']:.4f} ± {self.results['std_mae']:.4f}")
        
        print(f"\nBiological Metrics:")
        print(f"  Contact decay correlation: {self.results['mean_contact_decay_corr']:.4f} ± {self.results['std_contact_decay_corr']:.4f}")
        print(f"  Domain count difference: {self.results['mean_domain_diff']:.2f} ± {self.results['std_domain_diff']:.2f}")
        print("="*50)

def main():
    # Configuration
    model_path = "maskgit_highres.pt"  # Your trained model
    vae_path = "/scratch/rnd-rojas/Manan/baseResultsHighresolution/vae.49000.pt"
    genome_fasta = "/scratch/rnd-rojas/Manan/hg19.fa"
    
    # Load test data
    test_data_path = "/scratch/rnd-rojas/Manan/muse-maskgit-pytorch/"
    test_lowres = np.load(test_data_path + "lowres_dataset.npy")[-100:]  # Last 100 samples
    test_highres = np.load(test_data_path + "highres_dataset.npy")[-100:]
    test_coords = [tuple(c) for c in np.load(test_data_path + "hic_window_coords.npy", allow_pickle=True)[-100:]]
    
    # Convert to torch tensors
    test_images_lowres = [torch.from_numpy(img).float() for img in test_lowres]
    test_images_highres = [torch.from_numpy(img).float() for img in test_highres]
    
    # Initialize evaluator
    evaluator = HiCMapEvaluator(model_path, vae_path, genome_fasta)
    
    # Run evaluation
    results = evaluator.generate_and_evaluate(test_coords, test_images_lowres, test_images_highres)
    
    # Save and print results
    evaluator.save_results("evaluation_results.json")
    evaluator.print_summary()

if __name__ == "__main__":
    main() 