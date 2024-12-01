import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

# List of notebooks to execute in order
notebooks = ["hard_2_10.ipynb", "hard_2_20.ipynb", "hard_2_50.ipynb","hard_2_100.ipynb", "soft_2_10.ipynb", "soft_2_20.ipynb", "soft_2_50.ipynb", "soft_2_100.ipynb"]

output_dir = "executed_notebooks"
os.makedirs(output_dir, exist_ok=True)

for notebook in notebooks:
    print(f"Running {notebook}...")

    # Read the notebook
    with open(notebook, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Execute the notebook and capture outputs
    ep = ExecutePreprocessor(timeout=86400, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": "."}})

    # Save the executed notebook to a new file
    executed_notebook = os.path.join(output_dir, notebook)
    with open(executed_notebook, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Finished {notebook}. Outputs saved to {executed_notebook}.")