from hyperband import hyperband

def test_hyperband(objective_function, dict_config):
    hb = hyperband(R=9, nu=4, objective_function=objective_function, dict_config=dict_config)
    best_config = hb.main_loop()
    print(best_config)