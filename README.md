# AgenticSZZ: Replication Package

Replication package for the paper:

**"AgenticSZZ: Temporal Knowledge Graph-Guided Agentic Bug-Inducing Commit Identification"**

## Package Contents

```
├── agentic_szz/                   # Core implementation
│   ├── pipeline.py                # Main pipeline orchestrator
│   ├── kg/tkg.py                  # Temporal Knowledge Graph construction
│   ├── agent/
│   │   ├── bic_search_agent.py    # LLM agent for BIC search
│   │   └── graphiti_tools.py      # Graph traversal tools
│   ├── szz/blame.py               # Git blame analysis
│   ├── config/settings.py         # Configuration
│   └── utils/token_tracking.py    # Token usage tracking
│
├── evaluation/                    # Evaluation scripts
│   ├── run_evaluation.py          # Main evaluation (RQ1, RQ2, RQ3)
│   ├── rq0_run_experiments.py     # BIC category analysis (RQ0)
│   ├── rq0_analysis.py            # RQ0 figures and tables
│   ├── rq1_analysis.py            # RQ1 metrics
│   ├── sensitivity_analysis.py    # RQ3 sensitivity analysis
│   ├── rq3_tool_analysis.py       # RQ3 tool-use analysis
│   ├── create_pilot_sample.py     # Stratified sampling for sensitivity analysis
│   ├── generate_category_breakdown_table_v2.py  # RQ2 category table
│   ├── run_llm4szz_baseline.py    # LLM4SZZ controlled comparison
│   ├── statistical_tests.py       # McNemar's tests
│   └── tc_szz_baseline.py         # TC-SZZ evaluation
│
├── dataset/                       # Evaluation datasets (from LLM4SZZ)
│   ├── DS_LINUX.json              # 1,500 cases (Linux kernel)
│   ├── DS_APACHE.json             # 241 cases (Apache projects)
│   ├── DS_GITHUB.json             # 361 cases (GitHub projects)
│   └── git_clone_subject_repos.py # Download subject repositories
│
├── tests/                        # Leakage regression tests
├── SYSTEM_PROMPT.md               # Agent system prompt
└── pyproject.toml                 # Python dependencies
```

## Requirements

- Python 3.12+
- Neo4j (via Docker)
- Git

### Installation

```bash
pip install -e .
docker pull neo4j:latest
cp .env.example .env  # Set DEEPSEEK_API_KEY
python dataset/git_clone_subject_repos.py --dataset all
```

Run commands from the repository root. The evaluation runner manages its Neo4j containers. The default LLM is DeepSeek-V4-Pro with thinking disabled and temperature 1.0. Use `--model`, `--temperature`, and `--base-url` to select other configurations.

Experiment outputs are generated locally and excluded from Git. The illustrative example in `results/sample/` is included.

## Reproducing Results

### RQ0: Preliminary Study

```bash
python evaluation/rq0_run_experiments.py
python evaluation/rq0_analysis.py
```

### RQ1: Effectiveness Evaluation

```bash
python evaluation/run_evaluation.py \
    --dataset all --config full \
    --model deepseek-v4-pro --temperature 1.0 \
    --workers 16 --context-expand 2 \
    --output-dir results/rq1/running_1
python evaluation/rq1_analysis.py --results-dir results/rq1/running_1
```

The paper reports the mean of three independent runs. To reproduce it, repeat with `running_2` and `running_3`, then pass all three directories to `rq1_analysis.py` using `--runs`.

### RQ2: Ablation Study

```bash
for config in blame_only blame_fallback tkg_only; do
    python evaluation/run_evaluation.py --dataset all --config "$config" \
        --workers 16 --output-dir results/rq2
done
for config in agent_only full_no_graph; do
    python evaluation/run_evaluation.py --dataset all --config "$config" \
        --workers 16 --context-expand 2 --output-dir results/rq2/"$config"
done
# Without context expansion
python evaluation/run_evaluation.py --dataset all --config full \
    --workers 16 --context-expand 0 --output-dir results/rq2/no_expansion

# Category table, after running the LLM4SZZ baseline below
python evaluation/generate_category_breakdown_table_v2.py
```

The full approach reuses RQ1 run 1. `full_no_graph` presents the expanded candidates as a flat list without tools. `agent_only` uses blame candidates. Context expansion applies to `full` and `full_no_graph` only.

### RQ3: LLM Sensitivity Analysis

```bash
# DeepSeek-V4-Pro reuses RQ1 run 1.
python evaluation/run_evaluation.py --dataset all --config full \
    --model deepseek-flash --workers 16 --context-expand 2 \
    --output-dir results/rq3/deepseek-v4.1-flash

# Self-hosted models: gemma-4-31b, devstral-small-2, qwen3-coder-next.
# Repeat with each model name and matching output directory.
python evaluation/run_evaluation.py --dataset all --config full \
    --model gemma-4-31b --base-url <endpoint-url> \
    --workers 16 --context-expand 2 --output-dir results/rq3/gemma-4-31b

python evaluation/rq3_tool_analysis.py
```

Set `LOCAL_LLMS_API_KEY` if the self-hosted endpoint requires authentication.

### Candidate Limit Sensitivity

Generate the stratified sample after RQ0, then evaluate K=5, 10, 15, 20, 30, and 50:

```bash
python evaluation/create_pilot_sample.py --fraction 0.2 --seed 42 \
    --output results/pilot/pilot_cases_20pct.json
bash evaluation/run_sensitivity_analysis.sh
```

### Controlled Comparison (LLM4SZZ)

Download the [LLM4SZZ replication package](https://doi.org/10.6084/m9.figshare.27418236.v1) and follow the authors' setup and execution instructions. Use DeepSeek-V4-Pro for comparison with AgenticSZZ.

For the provided statistical analysis, place the three runs' per-case logs under `results/llm4szz/run_1/save_logs`, `run_2/save_logs`, and `run_3/save_logs`. The optional `run_llm4szz_baseline.py` wrapper supports a separately configured installation through `LLM4SZZ_DIR`.

### TC-SZZ and Statistical Tests

Download the [TC-SZZ replication package](https://figshare.com/articles/journal_contribution/SZZ-TSE23/23889792/2?file=46500478) and extract it to `vendor/tc-szz/`, containing `tse_new/rq12/json-output-raw/rq1/bic_tc_bugfix_commits_all.json`.

```bash
python evaluation/tc_szz_baseline.py
python evaluation/statistical_tests.py
```

The statistics script compares best and worst independent runs and the ablations using McNemar's exact test and Cohen's g. Individual comparisons can be selected with `--test main`, `--test robust`, `--test ablation`, or `--test tc_szz`.

### Additional Analyses

Optional scripts in `evaluation/` reproduce supporting measurements from the paper:

| Script | Analysis |
| :--- | :--- |
| `apache_failure_taxonomy.py` | Apache candidate and selection failures |
| `cross_file_ceiling.py` | BICs sharing no file with their fix |
| `scalability_analysis.py` | Graph size, runtime, memory, and tool calls |
| `confidence_calibration.py` | Prediction accuracy by reported confidence |

Run each script with `python evaluation/<script>.py` after generating RQ1 results and trajectories. Use the output directories shown above so the analysis scripts can find their inputs.

## Regression Tests

```bash
pip install -e '.[dev]'
python -m pytest tests/test_leakage_regression.py -q
```

These checks require no LLM calls or Neo4j. Checks using real commits require the Linux subject repository and skip when it is absent.
