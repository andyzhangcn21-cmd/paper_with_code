import argparse
import yaml
from src.experiments import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description='Run adversarial ICL experiments')
    parser.add_argument('--experiment', type=str, default='all',
                       choices=['1', '2', '3', 'all'],
                       help='Experiment to run')
    parser.add_argument('--config', type=str, default='configs/synthetic_config.yaml',
                       help='Path to config file')
    parser.add_argument('--save_results', action='store_true',
                       help='Save results to file')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize experiment runner
    runner = ExperimentRunner()
    
    # Run experiments
    if args.experiment in ['1', 'all']:
        print("\n" + "="*60)
        print("Running Experiment 1: Risk vs Radius")
        print("="*60)
        exp1_config = config['experiment1']
        runner.experiment1_risk_vs_radius(
            d=exp1_config['d'],
            m=exp1_config['m'],
            N=exp1_config['N']
        )
    
    if args.experiment in ['2', 'all']:
        print("\n" + "="*60)
        print("Running Experiment 2: Safe Radius vs Capacity")
        print("="*60)
        exp2_config = config['experiment2']
        runner.experiment2_safe_radius_vs_capacity(
            d=exp2_config['d'],
            N=exp2_config['N'],
            epsilon=exp2_config['epsilon']
        )
    
    if args.experiment in ['3', 'all']:
        print("\n" + "="*60)
        print("Running Experiment 3: Sample Complexity Tax")
        print("="*60)
        exp3_config = config['experiment3']
        runner.experiment3_sample_tax(
            d=exp3_config['d'],
            m=exp3_config['m'],
            N0=exp3_config['N0']
        )
    
    print("\nAll experiments completed!")


if __name__ == '__main__':
    main()