Recorded experiment
Model: Qwen/Qwen2.5-0.5B-Instruct
Revision: 7

Files
hallucination_lab.py          Complete application with all eight prompts
original_results.json        Original responses and run metadata
refined_results.json         Refined responses and run metadata
original_console.txt         Captured original execution output
refined_console.txt          Captured refinement execution output
original_runner_snapshot.py  Exact script used for the original phase
refined_runner_snapshot.py   Exact script used for the refinement phase
verify_checks.py             NumPy and metric checks
verification_checks.json     Actual output of those checks
requirements.txt             Main Python dependencies

1. Install the dependencies:
   python -m pip install -r requirements.txt
2. Run the two phases with new output filenames:
   python hallucination_lab.py --phase original --out my_original_results.json
   python hallucination_lab.py --phase refined --out my_refined_results.json
3. Optional independent checks:
   python verify_checks.py

The application refuses to overwrite an existing response JSON file. Keep
new results separate from the recorded results. The verification script
replaces verification_checks.json when it is run again.

