from pathlib import Path
from visualization import rank_results, top_n_solutions


def render_prompt(template_path="src/prompt_template.md", output_path="src/prompt.md", n_rank=15, n_solutions=3):
    """Render the prompt template by filling in rank_results and top_n_solutions."""
    template = Path(template_path).read_text()
    
    rank_output = rank_results(n=n_rank)
    code_list = top_n_solutions(n=n_solutions)
    
    # Format code solutions with ```python ... ``` blocks
    formatted_codes = []
    for code in code_list:
        if code.strip():
            formatted_codes.append(f"```python\n{code}\n```")
    
    top_n_formatted = "\n\n".join(formatted_codes) if formatted_codes else ""
    
    # Read dummy solution code
    dummy_path = Path("src/solutions/dummy.py")
    dummy_code = ""
    if dummy_path.exists():
        dummy_code = f"```python\n{dummy_path.read_text()}\n```"
    
    rendered = template.format(
        rank_results=rank_output,
        top_n_solutions=top_n_formatted,
        dummy=dummy_code
    )
    
    Path(output_path).write_text(rendered)
    return rendered


if __name__ == "__main__":
    render_prompt()
