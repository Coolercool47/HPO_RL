"""
Tests for run_experiment.py
Run with: pytest tests/test_run_experiment.py -v
"""
import pytest
import subprocess
import sys
import os
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports:
    """Test that all imports work correctly"""
    
    def test_import_run_experiment(self):
        """Test importing run_experiment module"""
        import run_experiment
        assert hasattr(run_experiment, 'AGENT_REGISTRY')
        assert hasattr(run_experiment, 'RECURRENT_ALGORITHMS')
        assert hasattr(run_experiment, 'generate_benchmark_hp_space')
        assert hasattr(run_experiment, 'collect_trajectory')
        assert hasattr(run_experiment, 'visualize_2d_trajectory')
        assert hasattr(run_experiment, 'run_experiment_with_visualization')
    
    def test_agent_registry(self):
        """Test that all agents are registered"""
        import run_experiment
        expected_agents = ['A2C', 'DQN', 'PPO', 'SAC', 'TD3', 'TRPO', 'MaskablePPO', 'RecurrentPPO']
        for agent in expected_agents:
            assert agent in run_experiment.AGENT_REGISTRY, f"Agent {agent} not in registry"
    
    def test_recurrent_algorithms(self):
        """Test recurrent algorithms set"""
        import run_experiment
        assert 'RecurrentPPO' in run_experiment.RECURRENT_ALGORITHMS


class TestCLI:
    """Test command-line interface"""
    
    def test_help_runs(self):
        """Test that --help works"""
        result = subprocess.run(
            [sys.executable, 'run_experiment.py', '--help'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        assert result.returncode == 0
        assert 'HPO-RL 2D Optimization' in result.stdout
        assert '--config' in result.stdout
        assert '--eval-mode' in result.stdout
    
    def test_invalid_agent(self):
        """Test that invalid agent is rejected"""
        result = subprocess.run(
            [sys.executable, 'run_experiment.py', '--agent', 'InvalidAgent', 
             '--config', 'configs/function_2d_test_ultra_low_budget.yaml'],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        assert result.returncode != 0
        assert 'Unknown algorithm' in result.stdout or 'InvalidAgent' in result.stdout


class TestHelperFunctions:
    """Test helper functions"""
    
    def test_generate_benchmark_hp_space(self):
        """Test HP space generation"""
        import run_experiment
        
        # Create mock backend
        class MockBackend:
            dimensions = 2
            bounds = [-5.0, 5.0]
        
        backend = MockBackend()
        hp_space = run_experiment.generate_benchmark_hp_space(backend)
        
        assert 'x0' in hp_space
        assert 'x1' in hp_space
        assert hp_space['x0']['continuous']['range'] == [-5.0, 5.0]
        assert hp_space['x1']['continuous']['range'] == [-5.0, 5.0]
    
    def test_get_ent_coef(self):
        """Test entropy coefficient getter"""
        import run_experiment
        
        # Test with float
        class MockAgent:
            ent_coef = 0.05
        
        agent = MockAgent()
        result = run_experiment._get_ent_coef(agent)
        assert result == 0.05
        
        # Test with default
        class MockAgent2:
            pass
        
        agent2 = MockAgent2()
        result2 = run_experiment._get_ent_coef(agent2)
        assert result2 == 0.01  # Default
    
    def test_get_learning_rate(self):
        """Test learning rate getter"""
        import run_experiment
        
        class MockAgent:
            learning_rate = 0.001
        
        agent = MockAgent()
        result = run_experiment._get_learning_rate(agent)
        assert result == 0.001


class TestIntegration:
    """Integration tests (require more time)"""
    
    @pytest.fixture
    def temp_logs_dir(self):
        """Create temporary logs directory"""
        temp_dir = tempfile.mkdtemp()
        original_dir = os.getcwd()
        yield temp_dir
        os.chdir(original_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.slow
    def test_short_training_run(self):
        """Test a very short training run (ultra-low budget)"""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, 'configs', 'function_2d_test_ultra_low_budget.yaml')
        
        # Skip if config doesn't exist
        if not os.path.exists(config_path):
            pytest.skip(f"Config not found: {config_path}")
        
        result = subprocess.run(
            [sys.executable, 'run_experiment.py', 
             '--config', 'configs/function_2d_test_ultra_low_budget.yaml',
             '--agent', 'PPO'],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=300  # 5 min timeout
        )
        
        # Print output for debugging
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
        
        assert result.returncode == 0, f"Training failed: {result.stderr}"
        assert 'Training' in result.stdout or 'Обучение' in result.stdout


class TestEvalMode:
    """Test evaluation mode logic"""
    
    def test_auto_mode_ppo(self):
        """Test that auto mode selects 'best' for PPO"""
        import run_experiment
        
        # PPO is not in RECURRENT_ALGORITHMS
        assert 'PPO' not in run_experiment.RECURRENT_ALGORITHMS
    
    def test_auto_mode_recurrent_ppo(self):
        """Test that auto mode selects 'final' for RecurrentPPO"""
        import run_experiment
        
        # RecurrentPPO is in RECURRENT_ALGORITHMS
        assert 'RecurrentPPO' in run_experiment.RECURRENT_ALGORITHMS


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

