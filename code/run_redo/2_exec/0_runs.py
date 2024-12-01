import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

# List of notebooks to execute in order
notebooks = ["soft_2_100.ipynb", "soft_4_100.ipynb", "soft_8_100.ipynb", "soft_16_100.ipynb", "soft_30_100.ipynb"]

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