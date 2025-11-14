def initialize_framework():

    from hpo_rl.core import factory
    from hpo_rl.models.simple_cnn import SimpleCNN
    from hpo_rl.trainers.torch_trainer import TorchTrainer
    from hpo_rl.backends.dummy import DummyBackend
    from hpo_rl.backends.real import RealTrainingBackend
    from hpo_rl.backends.table import TableBackend
    from hpo_rl.backends.function import OptimizationBenchmarkBackend
    from hpo_rl.environments.continuous_pipeline_env import ContinuousPipelineEnv
    from hpo_rl.environments.discretized_pipeline_env import DiscretizedPipelineEnv
    from hpo_rl.environments.masked_pipeline_env import DiscretePipelineEnv
    from hpo_rl.environments.lab_env import StrategyEnv

    factory.register_model("simple_cnn", SimpleCNN)

    factory.register_trainer("PyTorchTrainer", TorchTrainer)

    factory.register_backend("DummyBackend", DummyBackend)
    factory.register_backend("RealTrainingBackend", RealTrainingBackend)
    factory.register_backend("TableBackend", TableBackend)
    factory.register_backend("OptimizationBenchmarkBackend", OptimizationBenchmarkBackend)

    factory.register_env("DiscretePipelineEnv", DiscretePipelineEnv)
    factory.register_env("ContinuousPipelineEnv", ContinuousPipelineEnv)
    factory.register_env("DiscretizedPipelineEnv", DiscretizedPipelineEnv)
    factory.register_env("StrategyEnv", StrategyEnv)
