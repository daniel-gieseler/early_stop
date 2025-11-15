import importlib
from dataset import get_dataset
from task_evaluation import extrapolate_curves
import argparse
import json
from visualization import rank_results
from generation import render_prompt


def load_solution(solution_name: str) -> callable:
    module = importlib.import_module(f'solutions.{solution_name}')
    return getattr(module, solution_name)

def save_results(solution_name: str, results) -> None:
    output_path = f'src/results/{solution_name}.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

def evaluate_solution(
    solution_name: str,
) -> None:
    solution = load_solution(solution_name)

    curves = get_dataset(path='src/runs_experiments.json')

    epsilon = 0.05
    results = extrapolate_curves(curves, epsilon, solution)

    save_results(solution_name, results)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run loss prediction with specified predictor function')
    parser.add_argument('function_name', type=str, help='Name of the predictor function to import from predictor module')
    args = parser.parse_args()

    evaluate_solution(args.function_name)

    render_prompt()