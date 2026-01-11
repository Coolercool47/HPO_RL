from TPE import TPE

def test_TPE_hyp(objective_function, dict_config):
    FIXED_EPOCHS = 12
        
    target_func = lambda config: objective_function(config, r_i=FIXED_EPOCHS, dict_config=dict_config)

    gamma_func = lambda n: 0.25

    optimizer_tpe = TPE(
        objective_func=target_func, 
        N_init=3, 
        N_s=20, 
        budget=10, 
        dict_to_optimize=dict_config, 
        gamma_func=gamma_func
    )

    print("Starting TPE Optimization...")
    best_config, best_loss = optimizer_tpe.optimize()

    print("\n" + "="*30)
    print("Optimization Finished!")
    print(f"Best Loss: {best_loss:.4f}")
    print("Best Configuration:", best_config)
    print("="*30)