# AgenticSZZ: Replication Package

This is the replication package for the paper:

**"AgenticSZZ: Temporal Knowledge Graph-Guided Agentic Bug-Inducing Commit Identification"**

*Submitted to ISSTA 2026*

---

## Package Contents

```
├── agentic_szz/                   # Core implementation
│   ├── pipeline.py                # Main pipeline orchestrator
│   ├── kg/tkg.py                  # Temporal Knowledge Graph construction
│   ├── agent/
│   │   ├── bic_search_agent.py    # LLM agent with system prompt
│   │   └── graphiti_tools.py      # Graph traversal tools
│   ├── szz/blame.py               # Git blame analysis
│   ├── config/settings.py         # Configuration
│   └── utils/token_tracking.py    # Token usage tracking
│
├── evaluation/                    # Evaluation scripts
│   ├── run_evaluation.py          # Main evaluation (RQ1, RQ2)
│   ├── rq0_run_experiments.py     # BIC category analysis (RQ0)
│   ├── rq0_analysis.py            # Generate RQ0 figures/tables
│   ├── rq1_analysis.py            # Generate RQ1 metrics
│   ├── sensitivity_analysis.py    # Sensitivity analysis (Figure 4)
│   ├── run_llm4szz_baseline.py    # LLM4SZZ controlled comparison
│   └── create_pilot_sample.py     # Stratified sampling utility
│
├── dataset/                       # Evaluation datasets download from LLM4SZZ
│   ├── DS_LINUX.json              # 1,500 cases from Linux kernel
│   ├── DS_APACHE.json             # 241 cases from Apache projects
│   ├── DS_GITHUB.json             # 361 cases from GitHub projects
│   └── final_all_dataset.json     # Combined dataset (all cases)
│
├── results/sample/                # Example output format
│   └── example_output.json
│
├── AgenticSZZ_SYSTEM_PROMPT.md    # Agent system prompt
├── .env.example                   # Environment variables template
└── pyproject.toml                 # Python dependencies
```

---

## Requirements

- Python 3.12+
- Neo4j 5.x (via Docker)
- Git

### Installation

```bash
pip install -e .
docker pull neo4j:latest
cp .env.example .env  # Then edit with your API key
```

---

## Reproducing Results

```bash
# RQ0: Preliminary Study of BIC Category Distribution
python evaluation/rq0_run_experiments.py
python evaluation/rq0_analysis.py

# RQ1: Main Evaluation
python evaluation/run_evaluation.py --dataset all --config full --output-dir results/rq1
python evaluation/rq1_analysis.py

# RQ2: Ablation Study
python evaluation/run_evaluation.py --dataset all --config blame_only --output-dir results/rq2
python evaluation/run_evaluation.py --dataset all --config blame_fallback --output-dir results/rq2
python evaluation/run_evaluation.py --dataset all --config tkg_only --output-dir results/rq2
python evaluation/run_evaluation.py --dataset all --config agent_only --output-dir results/rq2
python evaluation/run_evaluation.py --dataset all --config full --output-dir results/rq2

# Sensitivity Analysis (in Threat to Validity)
python evaluation/sensitivity_analysis.py

# LLM4SZZ Controlled Comparison (in RQ2)
python evaluation/run_llm4szz_baseline.py --model deepseek
```
