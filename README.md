# 🧠 ChatGPT Prompt Engineering Dataset

## 🚀 Overview

Welcome to the **ChatGPT Prompt Engineering Dataset** — a curated collection of structured prompts designed for AI research, prompt engineering, and LLM training.

This dataset is built to help developers, researchers, and AI enthusiasts understand how prompts influence model behavior across different difficulty levels and domains.

It is designed for:

* AI training & fine-tuning
* Prompt engineering experiments
* Benchmarking LLM performance
* Educational AI research projects

> 💡 Think of this as a "gym for AI prompts" — where models learn how to think better.

---

## 🎯 Objectives

This project aims to:

* Organize high-quality prompts in a structured format
* Cover multiple domains (coding, reasoning, creativity, education)
* Provide multi-level difficulty (basic → advanced)
* Support AI evaluation and benchmarking systems
* Encourage experimentation in prompt engineering

---

## 📂 Dataset Structure

```
chatgpt-prompt-dataset/
│
├── data/
│   ├── prompts_basic.json
│   ├── prompts_intermediate.json
│   ├── prompts_advanced.json
│   ├── coding_prompts.json
│   ├── reasoning_prompts.json
│   ├── creative_prompts.json
│
├── notebooks/
│   ├── dataset_exploration.ipynb
│   ├── prompt_analysis.ipynb
│
├── src/
│   ├── loader.py
│   ├── cleaner.py
│   ├── evaluator.py
│
├── README.md
└── LICENSE
```

---

## 📊 Dataset Format

Each entry in the dataset follows a structured schema:

```json
{
  "id": 1,
  "category": "coding",
  "level": "intermediate",
  "prompt": "Write a Python function to simulate a basic neural network forward pass.",
  "expected_output_type": "code",
  "tags": ["python", "ai", "neural-network"]
}
```

---

## 🧩 Categories

### 🧠 Reasoning

Prompts that test logic, critical thinking, and structured reasoning.

Example:

* "If a system has 3 fail-safes and 2 of them fail, what is the risk model?"

---

### 💻 Coding

Prompts for generating code, debugging, and system design.

Example:

* "Build a Python API for a simple recommendation system."

---

### 🎨 Creative Writing

Prompts for storytelling, world-building, and imagination tasks.

Example:

* "Write a sci-fi story about AI controlling time loops in 2050."

---

### 📚 Education

Simplified explanations of academic concepts.

Example:

* "Explain quantum entanglement like I’m 12 years old."

---

### ⚙️ System Design / AI Behavior

Advanced prompts for designing AI agents and architectures.

Example:

* "Design an autonomous AI agent that manages stock trading decisions."

---

## 📈 Difficulty Levels

| Level        | Description                                             |
| ------------ | ------------------------------------------------------- |
| Basic        | Simple tasks, beginner-friendly prompts                 |
| Intermediate | Multi-step reasoning or coding required                 |
| Advanced     | Complex AI reasoning, system design, multi-domain tasks |

---

## 🧪 Example Entry (Advanced)

```json
{
  "id": 120,
  "category": "reasoning",
  "level": "advanced",
  "prompt": "Design a decentralized AI decision system for disaster response coordination in real-time environments.",
  "constraints": [
    "must be fault-tolerant",
    "must prioritize human safety",
    "must work under low connectivity conditions"
  ],
  "expected_output_type": "system_design",
  "tags": ["ai", "distributed-systems", "emergency-response"]
}
```

---

## 🧠 Use Cases

This dataset can be used for:

* Fine-tuning LLMs (ChatGPT-style models)
* Prompt engineering research
* AI agent development
* Kaggle competitions
* Academic AI experiments
* Benchmarking reasoning ability

---

## 📊 Planned Enhancements

Future upgrades include:

* 🔥 Prompt scoring system (AI evaluation)
* 🔥 RLHF-style preference dataset
* 🔥 Multi-language prompts
* 🔥 HuggingFace dataset conversion
* 🔥 Kaggle leaderboard integration

---

## ⚙️ Installation / Usage

Clone this repository:

```bash
git clone https://github.com/your-username/chatgpt-prompt-dataset.git
cd chatgpt-prompt-dataset
```

Load dataset in Python:

```python
import json

with open("data/prompts_basic.json", "r") as f:
    data = json.load(f)

print(data[0])
```

---

## 🧪 Example Analysis (Optional Notebook)

Run exploratory analysis:

```bash
jupyter notebook notebooks/dataset_exploration.ipynb
```

You can visualize:

* Prompt complexity distribution
* Category balance
* Output type frequency

---

## 🛡️ License

This project is released under MIT License.

You are free to:

* Use
* Modify
* Distribute
* Build upon

Just keep credit to original repository.

## 🌍 Vision

This project is part of a bigger vision:

> “To build structured intelligence through better prompts, not just bigger models.”

Because AI bukan cuma soal model besar — tapi soal **cara kita ngomong ke AI**.

## 🔥 Final Note

If you're building with this dataset:

* Think beyond templates
* Experiment with structure
* Push AI reasoning limits
