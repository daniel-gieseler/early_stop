import json

import importlib.util
import inspect
from pathlib import Path


def _get_sorted_metrics(results_dir="src/results"):
    """Calculate and return sorted list of (name, metric) tuples ranked by metric (10**avg_step)."""
    results_path = Path(results_dir)
    metrics = []
    
    for json_file in results_path.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
        
        steps = [item["step"] for item in data if "step" in item]
        if steps:
            avg_step = sum(steps) / len(steps)
            metric = 10 ** avg_step
            metrics.append((json_file.stem, metric))
    
    metrics.sort(key=lambda x: x[1])
    return metrics


def rank_results(results_dir="src/results", solutions_dir="src/solutions", n=None):
    """Read all JSON files in results folder and rank by metric (10**avg_step)."""
    solutions_path = Path(solutions_dir)
    metrics = _get_sorted_metrics(results_dir)
    
    # Limit to top n if specified
    if n is not None:
        metrics = metrics[:n]
    
    # Collect all data first to determine column widths
    data = []
    for name, metric in metrics:
        solution_file = solutions_path / f"{name}.py"
        description = ""
        if solution_file.exists():
            spec = importlib.util.spec_from_file_location(name, solution_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            func = getattr(module, name, None)
            if func and func.__doc__:
                description = inspect.cleandoc(func.__doc__)
        data.append((name, int(round(metric)), description))
    
    # Determine column widths (accounting for headers)
    rank_width = max(len("Rank"), len(str(len(metrics))))
    metric_width = max(len("Metric"), max(len(str(d[1])) for d in data) if data else 10)
    name_width = max(len("Name"), max(len(d[0]) for d in data) if data else 20)
    col_spacing = 2
    
    # Collect output lines
    output_lines = []
    
    # Header
    header = f"{'Rank':>{rank_width}}  {'Metric':>{metric_width}}  {'Name':<{name_width}}  Description"
    output_lines.append(header)
    output_lines.append("-" * (rank_width + metric_width + name_width + col_spacing * 3 + 20))
    
    # Rows
    for rank, (name, metric, description) in enumerate(data, 1):
        desc_lines = description.split('\n') if description else ['']
        first_line = True
        for line in desc_lines:
            if first_line:
                row = f"{rank:>{rank_width}}  {metric:>{metric_width}}  {name:<{name_width}}  {line}"
                output_lines.append(row)
                first_line = False
            else:
                row = f"{'':>{rank_width}}  {'':>{metric_width}}  {'':<{name_width}}  {line}"
                output_lines.append(row)
    
    # Print and return
    output = '\n'.join(output_lines)
    print(output)
    return output


def top_n_solutions(n, results_dir="src/results", solutions_dir="src/solutions"):
    """Return the Python code of the top n solutions ranked by metric (10**avg_step) as a list of strings."""
    solutions_path = Path(solutions_dir)
    metrics = _get_sorted_metrics(results_dir)
    # Exclude dummy solution
    filtered_metrics = [(name, metric) for name, metric in metrics if name != "dummy"]
    top_names = [name for name, _ in filtered_metrics[:n]]
    
    code_list = []
    for name in top_names:
        solution_file = solutions_path / f"{name}.py"
        if solution_file.exists():
            with open(solution_file, "r") as f:
                code_list.append(f.read())
        else:
            code_list.append("")
    
    return code_list

if __name__ == "__main__":
    print(top_n_solutions(3))