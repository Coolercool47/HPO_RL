from hpo_rl.experiments.run_experiment import run_n_experiments


config_HMM = {
    "full_args": {
        "algorithm": {
            "name": "HMM_MCMC",
            "budget": 500,             
            "n_init": 16,              
            "n_chains": 1,             
            "orchestrate_every": 10,    
            "T_mcmc": 0.001,            
            "sigma_fraction": 0.003,  
            "big_sigma_coef": 0.50,
            "temperature": 1.0,        
            "hmm_window": 10,           
            "hmm_obs_epsilon": 1e-8,
            "hmm_lambda_noise": 0.01, 
            "clone_noise": 0.05, 
            "burnin_fraction": 0.2
        }
    },
    "backend": {
        "name": "function", "function": "schwefel", "dimensions": 10, "noise_std": 100
    }
}

if __name__ == "__main__":
    run_n_experiments(config_HMM, n_experiments=3, inference_only=False)
