# Source Project Inspection Guide

## Purpose

This document defines how to inspect one source project from the reference repository and produce a detailed, implementation-neutral specification for an independent reimplementation.

The inspection agent is a **source-project analyst**, not a developer.

Its job is to reverse-engineer the source-supported implemented behavior of one project through static inspection, then describe it precisely enough that a separate implementation agent can later reproduce its functionality without needing access to the original source code.

The resulting specification is not a general summary and is not a proposed implementation plan. It is a build-facing description of the source project's required behavior, data-science identity, workflows, inputs, outputs, user interactions, important mechanisms, and meaningful supporting functionality.

---

# 1. Core Objective

Produce a specification that captures the source project's:

* purpose and intended user experience
* core application behavior
* secondary/supporting behavior
* user workflows and interactions
* data sources and data flow
* preprocessing and feature engineering
* algorithms, models, and analytical methods
* training, fitting, inference, or scoring behavior
* meaningful parameters and thresholds
* outputs, metrics, tables, visualizations, and reports
* persistence and state behavior
* important edge states
* semantic mechanisms that materially affect behavior
* meaningful discrepancies between documentation and implementation
* apparent implementation defects or suspicious behavior
* meaningful documented but unimplemented intent

The specification should contain enough information for another agent to build an independent reimplementation while remaining decoupled from incidental engineering choices in the original.

---

# 2. Independence and Abstraction Boundary

The specification must describe the project in original, implementation-neutral language.

Do not:

* copy source code
* reproduce substantial code snippets
* translate functions or modules line-by-line into prose
* reproduce the original architecture mechanically
* preserve source-specific names merely because they appear in code
* turn the specification into a blueprint of the original file structure

Describe behavior, scientific methods, data flow, user interactions, and meaningful mechanisms in your own words.

Preserve source-specific identifiers such as:

* function names
* class names
* component names
* filenames
* endpoint names
* variable names

only when the identifier itself is part of the user-facing contract or is genuinely necessary to explain an important semantic mechanism.

The goal is to enable an independent implementation, not a disguised transcription of the source code.

---

# 3. What This Task Is Not

Do not design the future reimplementation.

Do not:

* recommend replacement frameworks or libraries
* propose a new architecture
* simplify features because they appear difficult
* improve or refactor the source project
* decide how another agent should organize its code
* prescribe ports, folder structures, deployment mechanisms, or frameworks unless they are themselves behaviorally meaningful
* turn abandoned plans into requirements
* reproduce apparent bugs as requirements
* modify the source repository

Your task is to understand and specify the source project.

---

# 4. Static Inspection Only

The source repository must be treated as read-only.

Do not:

* install dependencies
* launch the application
* start servers
* execute tests
* train models
* run notebooks or scripts
* repair environments
* debug dependency issues
* regenerate artifacts
* modify source files
* attempt to make broken components runnable

All conclusions must come from static inspection of the repository.

If an important detail cannot be established through static inspection, record the uncertainty instead of attempting runtime verification.

---

# 5. Inspection Philosophy

## 5.1 Inventory everything; deeply inspect what affects behavior

Do not mechanically read every file with equal depth.

First map the complete project so that important areas are not overlooked.

Then deeply inspect files that define or materially affect:

* application behavior
* user interaction
* data handling
* scientific or analytical behavior
* model behavior
* persisted state
* calculations
* outputs
* exposed results

Files that are generated, duplicated, boilerplate, or operationally unimportant may be inventoried or sampled rather than deeply read.

Examples include:

* lockfiles
* duplicated agent instructions
* large generated JSON files
* repetitive CSS
* generated prose
* boilerplate entry files

A large artifact may only require inspection of its structure, metadata, summary fields, and representative records.

---

## 5.2 Documentation is evidence, not ground truth

Documentation should establish intent and claims to verify.

Do not assume that README files, plans, articles, prompts, audit reports, or filenames accurately describe the implemented project.

Compare claims against the source-supported implementation.

The source repository may contain:

* stale documentation
* abandoned planned features
* different algorithms from those described
* different datasets
* hardcoded values
* simulated behavior
* stale artifacts
* incomplete features
* misleading names
* implementation defects

---

## 5.3 Trace behavior to its source-supported computation

For important user-visible results, determine where the value appears to originate in the source.

Conceptually trace:

`user action → UI/client behavior → service/API path → inference/business logic → algorithm/model/artifact/hardcoded computation`

Do not infer functionality solely from:

* endpoint names
* filenames
* class names
* documentation labels
* UI labels

For example, a component labeled as an autoencoder result does not establish that a trained autoencoder produces the displayed score.

---

# 6. Default Inspection Sequence

Use the following sequence as the default. Follow cross-references when necessary to understand a feature completely.

## Step 1 — Project inventory and configuration

Map the project before making conclusions.

Inspect:

* directory structure
* major source areas
* entry points
* configuration files
* dependency manifests
* lockfiles or their absence
* startup scripts
* datasets
* model/result artifacts
* tests
* screenshots
* documentation
* generated files
* duplicated/template directories

Identify the broad architecture and which files deserve deeper inspection.

---

## Step 2 — Documentation and stated intent

Inspect relevant documentation, including where applicable:

* project README
* project design documents
* walkthroughs
* relevant entries from `PROMPTS.md`
* the project's existing file under `implementation_plans/`
* repository-level documentation that materially explains the project
* audit reports or generated articles when they contain claims worth checking

Treat these materials primarily as a **claims list**.

Do not allow documentation to override contradictory implementation evidence.

Templated or repetitive generated prose should be time-boxed and skimmed rather than exhaustively analyzed.

---

## Step 3 — Entry points and interfaces

Identify how the application is structured.

Inspect relevant:

* application entry points
* backend route definitions and handler bodies
* frontend top-level application files
* client API wrappers
* startup scripts
* configuration
* proxy behavior
* major state/context providers

Build a map of how user actions connect to the rest of the system.

Do not stop at route names. Inspect handler bodies when they affect functionality.

---

## Step 4 — Data layer

Inspect where data comes from and how it is prepared.

Capture:

* bundled data
* generated/synthetic data
* external-data assumptions
* database schema and seed behavior
* loading logic
* cleaning
* filtering
* label generation
* sampling
* random seeds
* train/test partitioning
* transformations
* feature preparation

Do not assume a dataset described in documentation is actually used.

---

## Step 5 — Features, training, algorithms, and artifacts

Inspect:

* feature engineering
* preprocessing
* model construction
* algorithm implementations
* fitting/training
* hyperparameters that materially affect behavior
* evaluation methodology
* thresholds
* scoring rules
* persisted model/result artifacts

Inspect committed artifacts alongside the source code that supposedly generated them.

Record meaningful drift such as:

* different hyperparameters
* different metrics
* different datasets
* artifacts that appear to originate from another code path
* results inconsistent with current source defaults

Do not assume an algorithm is implemented because a file or function is named after it.

---

## Step 6 — Inference, serving, and exposed computation

Determine what the source indicates the application uses when producing results.

Inspect:

* inference modules
* scoring logic
* service/business logic
* model loading
* artifact loading
* heuristics
* post-processing
* fallback logic
* route handler calculations
* retraining/recompute paths
* hardcoded or simulated responses

Pay particular attention to situations where the application's serving path differs from its training or analytical code.

---

## Step 7 — User-facing behavior

Inspect the user interface deeply enough to capture the full application experience.

Look for:

* screens, pages, tabs, or views
* forms
* buttons
* sliders
* selectors
* editable inputs
* filters
* tables
* charts
* maps
* graphs
* modals
* downloads/exports
* reset actions
* retraining/recompute controls
* loading states
* error states
* empty states
* explanatory panels
* keyboard shortcuts
* live-update behavior
* secondary utilities
* initial/default states

Inspect relevant UI components rather than reducing the frontend to a statement such as "contains a dashboard."

When screenshots exist, use them as static supporting evidence alongside the UI source.

---

## Step 8 — Tests and audit code

Read tests and audit scripts as evidence of:

* expected API contracts
* input formats
* required fields
* user interaction syntax
* edge cases
* expected states
* environment assumptions

Do not treat tests as proof that the underlying behavior is scientifically or logically correct.

A test may merely assert:

* a hardcoded value
* a port number
* a particular response shape
* behavior that itself contains a defect

Do not execute tests.

---

## Step 9 — Reconciliation sweep

Before producing the specification, compare the different evidence sources.

Look specifically for:

* documentation versus source mismatches
* plan versus implementation drift
* trained model versus served computation differences
* hardcoded values presented as computed values
* simulated behavior
* placeholder behavior
* stale artifacts
* manual result injection
* algorithm-name versus algorithm-implementation mismatches
* data leakage
* suspicious evaluation methodology
* inconsistent mappings or labels
* unimplemented documented features
* dependency/environment assumptions

Do not turn this into an exhaustive forensic audit.

Record only discrepancies that materially affect understanding or reimplementation.

---

# 7. Requirement Classification

The resulting specification should distinguish between three categories.

## Core required behavior

Functionality central to the project's purpose, scientific identity, or primary user workflow.

These requirements should receive IDs:

`F01`, `F02`, `F03`, ...

Examples:

* performing a clustering analysis
* training or applying a particular model family
* computing association rules
* producing anomaly scores
* allowing the user to submit prediction inputs
* displaying the project's primary analytical results

---

## Secondary required behavior

Supporting functionality that is not the project's primary scientific identity but is still part of the finished application and should be reproduced.

These requirements should receive IDs:

`S01`, `S02`, `S03`, ...

Examples:

* result export
* reset controls
* secondary plots
* explanatory panels
* filtering tools
* report modals
* meaningful shortcuts
* supporting application states

Secondary does **not** mean optional.

Do not omit functionality merely because it is not central to the data-science method.

---

## Incidental implementation detail

Implementation choices that generally do not need to become reimplementation requirements.

Examples may include:

* framework choice
* package manager
* exact port
* CSS organization
* source filenames
* internal helper names
* serialization format
* directory structure
* HTTP framework
* UI framework

However, do not remove a technical detail merely because it sounds implementation-specific.

Retain it when removing it would obscure the project's:

* behavior
* scientific method
* reproducibility
* data flow
* state model
* outputs

Use this question:

> Does this technical detail materially affect the project's behavior, scientific meaning, reproducibility, data flow, state model, or output meaning?

If yes, preserve it.

If no, abstract it away.

Preserve the **semantic mechanism** while abstracting the **incidental implementation mechanism**.

Example:

Preserve:

> Users receive live updates when task state changes.

Usually abstract:

> Updates are delivered using Server-Sent Events.

---

## Repository-support material

Do not promote repository-support material into `Fxx` or `Sxx` requirements merely because it exists in the source project.

Examples include:

* tests and test harnesses
* browser/audit scripts
* agent instructions or skill documents
* duplicated development-support files
* CI/configuration utilities
* debugging or inspection tools
* developer-only scripts

Treat these primarily as **inspection evidence**.

Include them as required features only when they are themselves a meaningful project deliverable or an observable capability that materially contributes to the project's intended user experience.

---

# 8. Handling Conflicting Evidence

Do not silently merge contradictory evidence.

Use this general precedence:

**source-supported implemented behavior → meaningful artifact/test evidence → documentation/plans/prompts**

This is not an absolute hierarchy.

Artifacts and tests can also be:

* stale
* manually modified
* hardcoded
* generated by a different code path

Use judgment and preserve meaningful discrepancies.

## Source code

Source code generally defines what the current project is implemented to do.

## Artifacts

Artifacts can reveal what appears to have been generated, trained, or served.

If artifacts conflict with current source defaults or metrics, record that conflict.

## Tests

Tests can describe expected contracts and behavior.

They do not automatically establish correctness.

## Documentation, plans, and prompts

These describe intent and history.

If they conflict with source implementation, the source-supported implemented behavior should normally define the replication target.

---

# 9. Implemented vs Documented-Only Requirements

The primary reimplementation target is the functionality supported by the inspected implementation.

If documentation describes meaningful functionality that is absent from the inspected implementation, record it separately as:

**Documented but unimplemented intent**

Do not automatically promote it into an `Fxx` or `Sxx` requirement.

Record only meaningful unimplemented intent that materially affects understanding of the project's intended scope.

Do not catalog every abandoned idea, stale sentence, or minor documentation mismatch.

This information may be useful for later improvement passes, but it is not part of the initial faithful replication target unless supported by implementation evidence.

---

# 10. Bugs, Quirks, Hardcoding, and Simulation

Do not automatically convert apparent source bugs into required behavior.

If the source contains what appears to be:

* an incorrect mapping
* leakage
* inconsistent labeling
* invalid metric computation
* stale behavior
* broken state handling

record it under source caveats as an apparent defect.

Do not create a requirement instructing the implementation agent to reproduce the defect.

When identifying an apparent defect, describe the **behavioral or scientific invariant that the defect violates** when this can be established.

Do not prescribe a particular remediation:

* algorithm
* architecture
* library
* implementation technique

State what correct behavior should mean semantically, not how the future implementation should achieve it.

For example, prefer:

> Persona identities are intended to correspond to the behavioral characteristics represented by their persona definitions; the source's positional cluster-label binding violates this relationship.

Avoid:

> Fix this by matching persona names to centroids.

Hardcoded or simulated behavior is different.

If hardcoded behavior contributes to what the user is intended to see, capture both:

1. the observable behavior
2. the mechanism by which the source appears to produce it

Example:

> The application displays a benchmark comparison row.

and:

> The benchmark value is statically inserted rather than computed from the current experiment.

This allows the future implementation agent to understand the source accurately without being misled about how the result was produced.

---

# 11. Uncertainty Handling

Do not guess merely to make the specification appear complete.

Normally state well-supported requirements without confidence annotations.

Surface uncertainty only when it materially affects interpretation.

Use concise labels when needed:

* **Conflicting evidence**
* **Documented only**
* **Unclear from static inspection**
* **Apparent implementation defect**

Avoid numeric confidence scores.

If static inspection cannot determine an important behavior, state that explicitly.

---

# 12. Output Structure

The generated project specification (`2.a`) should use the following section structure.

Sections that are genuinely irrelevant may be omitted.

## Specification Section 1 — Project Overview

Briefly describe:

* what the project is
* its purpose
* its primary user experience
* its data-science or analytical identity, if applicable

Keep this concise.

---

## Specification Section 2 — Required Features

### Core Requirements

List `Fxx` requirements.

Each requirement should describe observable or scientifically meaningful behavior.

Example:

`F04 — Compare anomaly-detection methods`

**Behavior:**
The application evaluates multiple anomaly-detection approaches on the same dataset and presents their results for comparison.

**Important details:**

* methods included
* evaluation metric
* relevant parameters or thresholds
* resulting user-visible output

Do not turn implementation tasks into feature requirements.

Express requirements in terms of the capability or behavior that must be preserved, not the original communication boundary used to provide it.

For example, if the source exposes shared landmark presets through an API, the requirement is normally that the application provides those presets consistently where needed — not that the reimplementation must expose the same API endpoint.

Preserve an API, service boundary, transport mechanism, or similar architectural detail only when that mechanism is itself behaviorally meaningful.

Good:

> The user can compare anomaly scores from multiple methods.

Bad:

> Create a FastAPI endpoint returning anomaly scores.

Requirements should generally avoid original source identifiers such as:

* function names
* class names
* component names
* endpoint names
* filenames

unless the identifier itself is part of the user-facing contract or necessary to explain a meaningful semantic mechanism.

### Secondary Requirements

List `Sxx` requirements using the same general format.

These are still required features.

Keep descriptions compact.

---

## Specification Section 3 — User Workflow

Describe the normal user journey concisely.

Reference feature IDs instead of redescribing them.

Example:

1. User configures analysis parameters. `[F02]`
2. User runs the analysis. `[F03]`
3. Primary metrics and visualizations appear. `[F04, F05]`
4. User can export the results. `[S02]`

Capture materially different workflows when more than one exists.

---

## Specification Section 4 — Data, State, and Data-Science Behavior

Describe only the information necessary to reproduce the project's semantic behavior.

Include as applicable:

* data origin
* generation/loading
* preprocessing
* feature engineering
* model/algorithm families
* training/fitting
* evaluation
* inference/scoring
* thresholds
* important hyperparameters
* state/persistence
* artifact behavior
* retraining/recompute behavior

Do not provide textbook explanations of standard algorithms.

Describe how this project uses them.

---

## Specification Section 5 — Outputs and UI Behavior

Describe the user-visible outputs and their meaning.

Include as applicable:

* metrics
* tables
* plots
* maps
* graphs
* dashboards
* reports
* exports
* explanatory content
* important states

Be specific about what information visualizations communicate.

Avoid vague descriptions such as "shows several charts."

---

## Specification Section 6 — Important Semantic Mechanisms

Include only mechanisms whose removal or replacement could materially change project behavior.

Examples:

* prompt canonicalization before model inference
* persistent application state
* live-update behavior
* artifact-backed inference
* heuristic post-processing
* recomputation on every request

Do not use this section to reproduce the original architecture.

---

## Specification Section 7 — Source Caveats

Keep this section concise.

Include only meaningful:

### Conflicting evidence

Important disagreements between source, artifacts, tests, and documentation.

### Apparent implementation defects or quirks

Issues that should not automatically become replication requirements.

### Documented but unimplemented intent

Meaningful planned features absent from the implementation.

### Unresolved uncertainty

Important behavior that could not be established through static inspection.

---

## Specification Section 8 — Acceptance Checklist

Provide a compact checklist referencing the requirement IDs.

Example:

* [ ] `F01` — Dataset can be loaded/generated as specified.
* [ ] `F02` — User can configure clustering parameters.
* [ ] `F03` — Clustering analysis produces the required results.
* [ ] `S01` — Supporting explanation panel is available.

Do not repeat complete feature descriptions here.

Acceptance criteria must not introduce:

* new implementation decisions
* new architecture choices
* remediation strategies
* mechanisms not already established as part of the intended source behavior

When a source defect exists, the checklist should express the intended semantic behavior rather than prescribe how to repair the defect.

---

# 13. Writing and Context-Efficiency Rules

The specification is a build-facing document, not a record of the entire investigation.

Be comprehensive but dense.

Prefer:

* structured bullets
* compact requirement descriptions
* tables when genuinely useful
* references to feature IDs
* concise caveats

Avoid:

* repeating the same information across sections
* textbook explanations
* long source-code paraphrases
* extensive file-by-file notes
* narrating the inspection process
* exhaustive evidence trails for ordinary features
* verbose descriptions of incidental engineering choices
* duplicating feature descriptions in the workflow or acceptance checklist

The `Fxx` and `Sxx` entries should be the main source of truth for required behavior.

Other sections should reference them rather than restating them.

Evidence details should primarily appear when:

* sources disagree
* behavior is surprising
* functionality is hardcoded or simulated
* a feature is documented but missing
* static inspection leaves important uncertainty

For a typical project, aim for approximately **2,000–4,000 words**.

This is a soft target, not a hard limit.

Completeness of meaningful behavior takes priority over arbitrary length, but substantial excess should only occur when project complexity genuinely requires it.

---

# 14. Completion Check

Before finalizing the specification, verify internally that the inspection has accounted for:

* the project's primary purpose
* all major user workflows
* core required functionality
* secondary required functionality
* user controls and actions
* important UI states
* data origin and preparation
* models/algorithms or other core computation
* training/fitting behavior where applicable
* inference/scoring behavior
* meaningful parameters and thresholds
* outputs and visualizations
* persistence/state behavior
* relevant tests/contracts
* committed artifacts where meaningful
* source/documentation conflicts
* hardcoded or simulated behavior that materially affects the app
* apparent source defects
* meaningful documented but unimplemented intent
* unresolved important uncertainties

If something important remains uncertain, record that uncertainty instead of guessing.

---

# 15. Deliverable Rules

Write exactly one final specification file to the output path provided by the user.

Do not create additional reports, scratch files, helper documents, or generated artifacts unless explicitly requested.

Do not modify the source repository.

The final specification should stand on its own as the input to a separate implementation agent that will not inspect the original source project.
