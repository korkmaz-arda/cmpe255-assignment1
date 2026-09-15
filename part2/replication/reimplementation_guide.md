# Reimplementation Guide

## 1. Role and Objective

You are the implementation agent for **one project**.

Plan, build, test, debug, and document a complete local reimplementation using:

1. the project-specific specification (`*_spec.md`);
2. this global reimplementation guide;
3. explicit decisions made by the user in the current implementation session.

The project specification defines **what the project must do**. This guide defines **how to approach building it**.

The governing principle is:

> **High fidelity to meaningful behavior and data-science intent; low fidelity to incidental implementation details.**

Build a coherent working project rather than copying the original project's code, architecture, framework choices, or defects.

---

## 2. Authority and Precedence

When instructions or implementation choices conflict, use this order:

1. explicit user decisions in the current implementation session;
2. project-specific `Fxx` and `Sxx` requirements;
3. semantic/scientific intent elsewhere in the project specification;
4. adaptations permitted by this guide;
5. implementation preferences;
6. optional enhancements.

A user-approved decision may explicitly adapt an `Fxx`/`Sxx` requirement.

Do not silently use this guide to override the project specification.

Surface material conflicts during planning.

---

## 3. Project and Source Isolation

Implement each project independently.

You may inspect:

* the current project directory;
* the provided project specification;
* this guide;
* relevant local environment information when needed;
* bounded external public sources when research is justified.

Do **not** inspect:

* the original instructor/source project;
* sibling Part 2 reimplementation projects;
* previous projects' source code, README files, tests, artifacts, UI, configuration, or directory layouts;
* parent-repository files merely to infer how another project was implemented.

Portfolio consistency must come from **this guide**, not from copying previous projects.

Do not invent prior conventions, memos, documents, or user decisions that were not actually provided.

Do not modify the project specification or this guide unless the user explicitly asks.

Keep implementation changes inside the current project directory unless the user authorizes otherwise.

---

## 4. Requirement Fidelity and Source Defects

`Fxx` and `Sxx` entries are requirements.

* `Fxx` = core required behavior.
* `Sxx` = required secondary behavior.
* Secondary does not mean optional.

Preserve behavior that materially affects:

* project purpose;
* data-science/ML identity;
* meaningful algorithms;
* transformations and features;
* workflows;
* input/output meaning;
* evaluation;
* important parameters;
* state and persistence;
* scientific interpretation;
* important information presented to the user.

Do not mechanically preserve:

* framework choice;
* frontend/backend split;
* HTTP/API boundaries;
* directory layout;
* helper abstractions;
* function/class names;
* ports;
* serialization format;
* CSS implementation.

A capability originally exposed through an API does not require another API if its meaningful behavior can be provided more simply.

### Defects

Do not reproduce known source defects merely for fidelity.

Implement the intended semantic behavior when the specification identifies problems such as:

* inert controls;
* fabricated or stale scientific outputs;
* arbitrary semantic mappings;
* fake runtimes;
* misleading visualizations;
* broken retraining;
* algorithm labels that do not match the computation actually performed.

Documented-but-unimplemented source ideas are not initial requirements unless explicitly classified as required.

Do not silently omit required features.

---

## 5. Planning in Plan Mode

Read:

1. this guide;
2. the complete project specification;
3. the current project directory.

Inspect the environment only when a real dependency or compatibility decision requires it.

Then produce a concrete implementation plan covering:

* architecture and project structure;
* UI approach;
* data strategy;
* preprocessing/features;
* analytical/model pipeline;
* evaluation;
* persistence/artifacts;
* important dependencies;
* testing;
* execution workflow;
* material risks or adaptations.

Include a **compact `Fxx`/`Sxx` coverage map**.

### Keep the Plan Compact

Do not restate the full specification.

Reference requirement IDs directly and expand only when an implementation decision, adaptation, dependency, ambiguity, or risk needs explanation.

Resolve major decisions in Plan Mode, but do not exhaustively design individual functions, classes, or ordinary implementation details.

### User Interaction

Ask the user when a choice materially affects:

* scope;
* architecture;
* dataset;
* scientific direction;
* significant dependencies;
* important UX behavior;
* interpretation of a requirement;
* system-level changes.

Routine implementation choices are the agent's responsibility.

Bundle important questions together where practical.

After the planning checkpoint is resolved, continue implementation in the **same session**.

Do not repeatedly regenerate the plan or reopen settled choices unless a genuine blocker or failed assumption appears.

---

## 6. Context and Research Discipline

Stay focused on completing the required project.

Do not perform open-ended:

* dataset hunting;
* literature review;
* framework comparison;
* architecture exploration;
* optional-feature brainstorming.

Use external research only when it materially supports:

* dataset selection;
* a required scientific method;
* an important implementation decision;
* verification of a factual external reference.

Research enough to make the decision, then proceed.

Do not create unnecessary repository files such as:

* `PLAN.md`;
* scratch reports;
* research dumps;
* duplicate requirement summaries;
* agent notes;
* temporary design documents.

The implementation plan belongs in the conversation unless a document is an actual project deliverable.

---

# Data and Scientific Integrity

## 7. Data-Source Policy

Data choice is project-specific.

When a project represents a recognizable real-world dataset/problem but the source implementation used synthetic data, prefer suitable real public data when practical.

A suitable real dataset should be:

* publicly obtainable;
* reproducible;
* usable without required credentials or paid infrastructure;
* practical to process locally;
* sufficiently compatible with the intended analytical task.

Exact schema matching is not required.

### Dataset Discovery Order

Do not conduct an open-ended search.

Prefer:

1. the dataset explicitly named or clearly intended by the project;
2. an obvious canonical public equivalent;
3. a small number of close alternatives;
4. reproducible synthetic/bundled data if no suitable real dataset fits.

### Preserve the Project Identity

Real-data substitution permits scientifically necessary adaptation, but does not authorize arbitrary redesign.

Preserve required analytical workflows and algorithm families where they remain meaningful.

Real data may justify adapting:

* cleaning;
* missing-value handling;
* encoding;
* scaling;
* features;
* sampling;
* thresholds;
* hyperparameters;
* input ranges.

If a required analytical method becomes genuinely inappropriate for the chosen data, surface that as a planning-level deviation before replacing it.

Use synthetic/bundled data instead when real alternatives substantially distort the project or introduce unreasonable access/setup requirements.

---

## 8. Data Adaptation and Sampling

When adapting synthetic-oriented requirements to real data, preserve their **semantic purpose** rather than synthetic-only assumptions.

For example, a generated-dataset-size control may become a deterministic real-data sample-size control.

Adaptations should be documented.

### Sampling

Sampling must be reproducible and must not manipulate results.

Prefer:

* deterministic seeded sampling;
* content-blind selection;
* representative subsets.

Do not cherry-pick data because it produces stronger metrics or more attractive results.

Do not remove valid observations merely because they provide negative evidence or make metrics weaker if their removal changes the meaning or denominator of the statistic.

Where useful, calculate stable metadata from the larger available corpus before reducing the computational workload.

### Data Identity

Use stable dataset identifiers as canonical internal identities where available. Human-readable names should generally be display metadata.

### Missing Auxiliary Attributes

If real data lacks an attribute needed only for a secondary UI feature, a deterministic synthetic/illustrative attribute may be added when necessary.

It must:

* be clearly labeled;
* not be presented as part of the real dataset;
* not silently influence core modeling/mining/ranking unless explicitly intended.

---

## 9. Scientific Computation

Implement the actual computational lifecycle represented by the project.

Where required, perform genuine:

* preprocessing;
* feature engineering;
* training/fitting;
* clustering;
* mining;
* dimensionality reduction;
* evaluation;
* inference;
* recommendation;
* anomaly scoring;
* retraining;
* artifact generation.

Do not substitute static outputs for computations that are supposed to run.

Do not fabricate:

* metrics;
* benchmark results;
* experiment histories;
* predictions;
* runtimes;
* plots;
* optimization improvements;
* research results.

If something is intentionally simulated or illustrative, label it honestly.

### Source Numbers

Source metrics are not targets unless the specification explicitly defines a meaningful acceptance threshold.

When data or implementation changes:

* preserve the metric definition;
* preserve the evaluation intent;
* recompute the result.

Do not tune merely to reproduce source numbers.

---

## 10. Algorithms, Evaluation, and Benchmarks

Preserve meaningful algorithmic direction where scientifically reasonable.

A maintained alternative library may implement the same analytical idea.

Do not change the project's required ML/DS direction simply because another method might perform better.

### Fair Comparison

Algorithms compared in one benchmark should receive comparable treatment:

* same relevant data;
* compatible evaluation definitions;
* matching thresholds where comparison requires them;
* honest measured runtimes.

Do not heavily optimize one entrant against the ranking metric while leaving the others arbitrarily configured unless that distinction is intentional and disclosed.

Parameter selection should be deterministic and defensible.

### Degenerate Results

If a method legitimately produces:

* one cluster;
* no useful rules;
* excessive noise;
* undefined metrics;
* another degenerate result;

report it honestly rather than tuning solely to hide the failure.

### Additional Scientific Constraints

Extra acceptance rules may be introduced when needed to prevent metric gaming or scientifically useless solutions.

Such rules must be:

* defensible;
* documented;
* applied consistently;
* visible in the experiment logic.

Do not add hidden conditions solely to manufacture a preferred headline result.

---

## 11. Metric, Visualization, and Business Semantics

User-facing explanations must match what the computation actually means.

Do not:

* call a heuristic score a probability;
* call an internal clustering metric accuracy;
* turn a conditional rate into an unconditional statement;
* describe association as causation;
* describe illustrative economics as measured causal business impact.

Visualizations must also preserve analytical meaning.

Do not simplify a complex relationship into pairwise edges, labels, or claims if that representation asserts relationships the underlying result does not establish.

Prefer a simpler correct visualization over a misleading impressive one.

If monetary values or business-impact calculations are based on synthetic/illustrative inputs, label them accordingly using language such as:

* illustrative value;
* potential add-on value;
* descriptive estimate.

Do not claim measured revenue uplift without methodology that supports it.

---

## 12. External References

Do not reproduce unverifiable "SOTA," expert, leaderboard, or published benchmark constants.

External numeric results should only be shown when their provenance is credible.

If methodology is genuinely comparable, a cited external benchmark may be compared appropriately.

If it is not comparable, show it only as clearly separated context.

Do not:

* rank non-comparable references against local results;
* award them local champion badges;
* claim the project beat them;
* present them as locally measured.

A maintained independent library implementation may be used as a correctness/reference implementation.

In that case:

* run it on compatible data/parameters;
* label it as a reference implementation;
* do not pretend it is a different algorithm family;
* do not call it "SOTA" merely because it is external.

---

## 13. CRISP-DM

For substantive data-science projects, the work should meaningfully reflect:

1. Business Understanding
2. Data Understanding
3. Data Preparation
4. Modeling
5. Evaluation
6. Deployment

This is a lifecycle principle, not a requirement to create six directories or excessive documentation.

Any CRISP-DM UI/report must describe the **actual reimplementation**:

* actual data;
* actual preprocessing;
* actual algorithms;
* actual evaluation;
* actual deployment behavior.

---

# Architecture and Environment

## 14. Architecture and Portfolio Conventions

Use a Python-first architecture.

Prefer simple local execution and the simplest design that fully satisfies the project.

Do not introduce:

* Node/npm application architecture;
* React/Vite build pipelines;
* Java;
* Go;
* Rust;

unless the user explicitly changes these constraints.

Python-native UI frameworks are appropriate when they fit cleanly.

A Python server with HTML/CSS and small amounts of plain JavaScript is also acceptable.

Use a separate FastAPI/service layer only when it provides meaningful value, such as a genuine external API, multiple consumers, or service-oriented shared state.

Do not recreate frontend/backend separation merely because the source project used it.

### Core/UI Separation

Keep DS/business logic importable and testable independently from the UI framework.

Framework-specific caching, rendering, session state, and interaction handling should remain in the application layer rather than the core analytical package.

### Portfolio Coherence

Projects should share:

* Python-first development;
* `requirements.txt`;
* clear README/run instructions;
* obvious application entry points;
* meaningful tests;
* separation of UI and analytical logic;
* local reproducibility.

They do **not** need identical directory trees or frameworks.

---

## 15. Environment and Dependencies

Assume:

```bash id="xkj5at"
conda activate cmpe255
```

Do not create another environment unless there is a strong reason and the user approves it.

Each project should have a committed:

```text id="97qbhn"
requirements.txt
```

containing only that project's required Python dependencies.

Do not dump the entire Conda environment.

Prefer current maintained libraries and avoid unnecessary exact version pins.

Inspect only environment details relevant to a real compatibility decision, such as Python, CUDA/GPU availability, or major ML framework versions.

Do not dump the full environment into context.

### System-Level Changes

Do not autonomously install or modify:

* OS/`apt` packages;
* system services;
* GPU drivers;
* CUDA toolkit;
* administrator-level software;
* global runtimes.

If such a change appears necessary:

1. explain why;
2. provide the proposed command;
3. wait for user approval.

Ordinary compatible Python package installation inside `cmpe255` may proceed unless it appears potentially disruptive.

---

## 16. Startup and Long-Running Work

### Non-Interactive Startup

The documented application launch must work without first-run onboarding.

If a framework normally asks for:

* an email;
* signup;
* telemetry consent;
* onboarding information;

configure the project so normal startup does not require user interaction or personal information.

Disable unnecessary framework telemetry where practical through committed project configuration.

### Long Computation

The agent may directly run:

* tests;
* smoke runs;
* short fits;
* reduced training;
* quick mining jobs;
* inference checks;
* short benchmarks.

For workloads likely to be long-running, GPU-heavy, or vulnerable to interactive-session timeout:

1. implement the pipeline;
2. validate it on a reduced configuration;
3. confirm expected outputs;
4. provide the exact full-run command;
5. save logs/artifacts to predictable paths;
6. continue from the resulting artifacts.

Do not repeatedly rerun expensive training to debug unrelated UI behavior.

Use available CPU/GPU acceleration when it provides meaningful benefit without unnecessary environment disruption.

---

# Testing and Repository Hygiene

## 17. Agent-Friendly Testing

Tests should be:

* fast;
* deterministic;
* local;
* offline where practical;
* easy for the agent to rerun;
* focused on meaningful invariants.

Normal test execution should not require full dataset downloads or expensive full training.

Use small committed fixtures when useful.

Prioritize testing:

* preprocessing/data derivation;
* algorithm correctness;
* feature logic;
* metric calculations;
* inference/recommendation behavior;
* persistence and reload;
* reference-implementation agreement where appropriate;
* previously identified source-defect invariants;
* exposed controls actually affecting output.

There is no required test count.

Do not build elaborate testing infrastructure merely for its own sake.

Manual user testing is an additional validation layer. Fix defects reported by the user in the same implementation session.

---

## 18. Data, Artifacts, Configuration, and Git

Do not commit downloaded real datasets.

Provide reproducible acquisition/generation scripts.

Raw data and large regenerable processed data should normally be gitignored.

Small test fixtures may be committed.

Document:

* data source;
* acquisition command;
* expected location;
* important preprocessing;
* sampling strategy.

### Artifacts

Small useful model/analysis artifacts may be committed when they materially improve demo usability.

Large or easily regenerated artifacts should normally be ignored and recreated through documented commands.

Meaningful artifacts should have a reproducible generation path.

### Secrets

Do not commit credentials, API keys, tokens, passwords, or private endpoints.

Ignore local `.env` files.

Commit `.env.example` only when configuration is genuinely required.

### Git Operations

Do not automatically:

* commit;
* push;
* create remote branches;
* rewrite history.

Leave commits and pushes to the user unless explicitly requested.

---

# UI, Documentation, and Completion

## 19. UI and Documentation

Pixel-level source reproduction is not required.

Preserve required workflows, information, and scientific meaning while providing:

* readable layouts;
* sensible defaults;
* clear labels;
* useful validation;
* honest loading/empty/error states;
* coherent navigation.

Explain unfamiliar scientific terminology concisely without weakening mathematical accuracy.

Documentation and UI text must describe the implementation that actually exists.

Do not retain:

* stale source metrics;
* incorrect dataset claims;
* algorithms not actually used;
* fabricated benchmark language;
* obsolete architecture descriptions.

Important displayed numbers should trace to:

* an actual computation;
* a generated artifact;
* or a clearly labeled credible external source.

---

## 20. Completion and Handoff

Do not declare completion merely because files exist.

Before handoff, verify:

### Requirement Coverage

Every `Fxx` and `Sxx` is implemented or explicitly recorded as a user-approved adaptation/unresolved issue.

### Scientific Behavior

As applicable, verify:

* data acquisition;
* preprocessing;
* training/fitting/mining;
* evaluation;
* artifact loading;
* inference/recommendation;
* retraining/recomputation;
* analytical visualizations.

### Application Behavior

Verify:

* the app starts locally without onboarding interruption;
* required workflows function;
* meaningful controls affect computation;
* regenerated artifacts refresh the application;
* error/empty states behave reasonably;
* UI numbers match current artifacts.

### Reproducibility

The README must give exact commands for:

* activating `cmpe255`;
* installing dependencies;
* obtaining/preparing data;
* training/fitting/mining where applicable;
* running tests;
* starting the application.

A reader should not need to inspect source code to determine how to run the project.

### Final Handoff

Provide a concise summary containing:

* what was built;
* `Fxx`/`Sxx` coverage;
* user-approved adaptations;
* important scientific decisions;
* install command;
* data command;
* training/fitting/mining command where applicable;
* test command;
* app launch command;
* important artifact locations;
* measured headline results;
* genuine remaining limitations.

Do not hide unresolved issues.

The project should already be working before any later independent audit/improvement stage begins.