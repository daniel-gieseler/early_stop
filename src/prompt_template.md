# Role

- You are doing one iteration in an optimization loop.
- You can see the history of previous solution.
- Implement a solution that is both novel and introduces the most impactful yet simplest change.
- Novel, impactful and simple.
- You can build on top of previous solutions.
- We are gradually exploring the space of solutions. So trying one simple idea at a time, helps us identify what actually makes a difference.

# Solution

- Think deeply about the problem.
- You don't need to open any other files to look for more information.
- Choose a solution_name.
- Create a solution_name.py at the /solutions folder.
- In that script, at least one function must be called solution_name, it will be the entry point for the evaluator.
- This function must follow the contract:

{dummy}

- The input is a loss curve, a list of pairs (step, loss)
    - the curve is monotonically decreasing because it is actually the cumulative minimum loss.
    - steps and loss are already converted to log-log space.
    - points on the curve have been subsampled to be uniform in log-space.
    - the curve is incomplete - it does not contain the last part of the training.

# Evaluation

- We are trying to model the last stage of training, where the loss converges in a predictable way.
- The function you will make models the first derivative of the loss once it starts converging.
- The function will be used to extrapolate the rest of the training.
- The function will be evaluated by how early can it correctly predict the final loss within a certain error.
- The function will be run on several curves, the final evaluation metric is the average step of early stop.  
- The sooner, the better. That is we want to minimize the average step.

# Rank
{rank_results}

# Best Examples
{top_n_solutions}

