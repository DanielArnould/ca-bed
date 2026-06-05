# CA-BED [ICLR 2026 Workshop]

**CA-BED: Conversation-Aware Bayesian Experimental Design**

[![arXiv](https://img.shields.io/badge/arXiv-2606.01182-b31b1b.svg)](https://arxiv.org/abs/2606.01182)

Large Language Models (LLMs) excel at static reasoning tasks, yet their perfor
mance often degrades in interactive scenarios where information must be actively
acquired through questioning. A key challenge lies in selecting questions that re
duce uncertainty while incorporating responses that may be ambiguous or only
partially informative. To address this, we propose Conversation-Aware Bayesian
Experimental Design (CA-BED), an inference-time probabilistic dialog planning
framework that integrates Bayesian Experimental Design with LLM-based likeli
hood estimation to optimize question selection over multiple conversational turns.
CA-BED maintains a belief distribution over hypotheses, anticipates possible an
swers, and propagates expected information gain through a simulated conversa
tion tree. Across two structured entity-deduction benchmarks, CA-BED yields an
average 21.8\% improvement in success rates over direct prompting, with compa
rable gains relative to alternative information-seeking methods. It achieves these
gains with an average increase of only 1.8 conversational turns compared to di
rect prompting. These results suggest that probabilistic conversation planning is
a promising direction for interactive reasoning in structured information-seeking
settings.

## Setup

1. Install [pixi](https://pixi.prefix.dev/latest/)
2. Clone the `ca-bed` package
3. Run `pixi install` in the root directory
4. Create a `.env` file and populate it with the following variables:
   ```env
   API_KEY=your_api_key_here
   API_BASE_URL=your_api_base_url_here
   MAX_CONCURRENT_REQUESTS=10
   ```

## Use

Run experiments with `pixi run experiments <task_name> [options]`, which implements CA-BED, CA-BED + Answer-Planning, UoT, and Direct Prompting.

### Available Tasks

[**Detective Cases**](https://github.com/tmlr-group/AR-Bench)

| Task Name                  | Description                                    |
| -------------------------- | ---------------------------------------------- |
| `detective_direct`         | Direct prompting baseline (no reasoning tree). |
| `detective_uot`            | Uncertainty of Thoughts (UoT)                  |
| `detective_bayesian`       | CA-BED                                         |
| `detective_bayesian_multi` | CA-BED + Answer-Planning                       |

[**Twenty Questions**](https://github.com/zhiyuanhubj/UoT)

| Task Name                | Description                                    |
| ------------------------ | ---------------------------------------------- |
| `twentyq_direct`         | Direct prompting baseline (no reasoning tree). |
| `twentyq_uot`            | Uncertainty of Thoughts (UoT)                  |
| `twentyq_bayesian`       | CA-BED                                         |
| `twentyq_bayesian_multi` | CA-BED + Answer-Planning                       |

### Common Arguments

| Argument                   | Type  | Default               | Description                                          |
| -------------------------- | ----- | --------------------- | ---------------------------------------------------- |
| `--task`                   | `str` | _required_            | The specific task to run.                            |
| `--experiment_name`        | `str` | `run_<timestamp>`     | Name of the experiment directory for saving results. |
| `--seed`                   | `int` | `42`                  | Random seed for dataset sampling.                    |
| `--questioner_model`       | `str` | `"deepseek-chat"`     | Model key for the questioner.                        |
| `--answerer_model`         | `str` | `"deepseek-reasoner"` | Model key for the answerer.                          |
| `--start_idx`              | `int` | `0`                   | Starting index for dataset sampling.                 |
| `--end_idx`                | `int` | `10`                  | Ending index for dataset sampling.                   |
| `--max_conversation_depth` | `int` | `20`                  | Maximum conversation depth.                          |
| `--max_concurrent_tasks`   | `int` | `6`                   | Maximum concurrent tasks to run.                     |

### Additional Arguments for Tree-Based Methods

| Argument                 | Type    | Default | Description                                             |
| ------------------------ | ------- | ------- | ------------------------------------------------------- |
| `--max_question_nodes`   | `int`   | `3`     | Maximum number of question nodes per turn.              |
| `--max_lookahead_depth`  | `int`   | `2`     | Lookahead search depth for planning.                    |
| `--confidence_threshold` | `float` | `0.8`   | Confidence threshold for terminating early.             |
| `--estimator_confidence` | `float` | `0.7`   | Ɛ confidence constant for the LLM likelihood estimator. |

## Evaluation and Analysis

After running one or more experiments, you can evaluate and compare their results using `pixi run analysis <path to experiment dir>`. This reads the files generated by each experiment, computes performance statistics, and summarises them in a comparison table.

### Summary Statistics Explained

Each experiment directory is evaluated independently. For each run, the following metrics are computed:

Here is the markdown for that table:

| Metric              | Description                                                       |
| ------------------- | ----------------------------------------------------------------- |
| Top-1               | Whether the model's most likely guess matches the correct answer. |
| Top-3               | Whether the correct answer appears in the top-3 guesses.          |
| Conversation Length | Number of question-answer turns in the dialogue.                  |
| Start / End Time    | Used to measure total runtime duration.                           |
| Token Usage         | Input/output token counts for both questioner and answerer.       |

### Arguments

| Argument       | Type   | Default    | Description                                                |
| -------------- | ------ | ---------- | ---------------------------------------------------------- |
| `-p`, `--path` | `Path` | _required_ | Base experiment directory containing method subdirectories |
