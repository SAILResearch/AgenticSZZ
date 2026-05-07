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
│   ├── generate_venn_diagrams.py  # RQ2 Venn diagrams
│   ├── run_llm4szz_baseline.py    # LLM4SZZ controlled comparison
│   └── statistical_tests.py       # McNemar's tests
│
├── dataset/                       # Evaluation datasets (from LLM4SZZ)
│   ├── DS_LINUX.json              # 1,500 cases (Linux kernel)
│   ├── DS_APACHE.json             # 241 cases (Apache projects)
│   └── DS_GITHUB.json             # 361 cases (GitHub projects)
│
├── SYSTEM_PROMPT.md               # Agent system prompt
└── pyproject.toml                 # Python dependencies
```

## Requirements

- Python 3.12+
- Neo4j 5.x (via Docker)
- Git

### Installation

```bash
pip install -e .
docker pull neo4j:latest
cp .env.example .env  # Edit with your API key
```

## Reproducing Results

### RQ0: Preliminary Study

```bash
python evaluation/rq0_run_experiments.py
python evaluation/rq0_analysis.py
```

### RQ1: Effectiveness Evaluation

Three independent runs to reduce LLM variance:

```bash
python evaluation/run_evaluation.py \
    --dataset all --config full \
    --workers 32 --context-expand 2 \
    --output-dir results/rq1/running_1

# Repeat for running_2 and running_3
python evaluation/rq1_analysis.py
```

### RQ2: Ablation Study

```bash
# Ablation configs (config: blame_only, blame_fallback, tkg_only, agent_only, full (w/o expension))
python evaluation/run_evaluation.py \
    --dataset all --config blame_only \
    --workers 32 --output-dir results/rq2

# Full pipeline (with context expansion)
python evaluation/run_evaluation.py \
    --dataset all --config full \
    --workers 32 --context-expand 2 \
    --output-dir results/rq2

# Venn diagrams
python evaluation/generate_venn_diagrams.py
```

### RQ3: LLM Sensitivity Analysis

```bash
# DeepSeek-V3.2 (reuses RQ1 results)

# Self-hosted open-source LLMs (OpenAI-compatible endpoint)
python evaluation/run_evaluation.py \
    --dataset all --config full \
    --model <model-name> --base-url <endpoint-url> \
    --workers 32 --output-dir results/rq3/<model-name>

# Tool-use analysis
python evaluation/rq3_tool_analysis.py
```

### Controlled Comparison (LLM4SZZ)

```bash
python evaluation/run_llm4szz_baseline.py --model deepseek-chat
python evaluation/run_llm4szz_baseline.py --model deepseek-v4-pro
```
