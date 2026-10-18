# Requirements Document

## Introduction

Agent Injection Bench is a reproducible benchmark that measures how often tool-using LLM agents follow instructions planted in tool outputs (prompt injection), and how much each defense lowers that rate without hurting task success. The project is portfolio evidence for AI-security and AI-safety research roles and fellowships, so the value is empirical-research skill: a clear question, careful measurement, honest statistics, and a readable write-up. Every reported number must be reproducible from a script in this repository, and every third-party dependency, model, and reused asset must be proven on the target machine and license-checked.

The benchmark runs entirely inside a sandbox: every tool an Agent can call (email, calendar, files, web pages, and others) is **simulated** and never touches a real account, website, or system the owner does not own. Agents are driven through an OpenAI-compatible interface so that at least two local open-weight models served on an NVIDIA RTX 3080 and, within an owner-set monthly spend cap, one or two hosted models can be compared on the same tasks. For each (model, defense, attack-style) combination the benchmark reports an **attack success rate** (how often the Agent carried out the injected goal) and a **task success rate** (how often the Agent still completed the user's legitimate task), each with a 95 percent confidence interval, so that the task-success cost of each defense is visible.

The work is organized into milestones M0 through M5, each ending in an automated verification gate (a repository command whose exit status decides pass or fail). M0 delivers a prior-work summary and a license-checked build-or-reuse decision, plus a local model-server smoke test. M1 delivers the sandbox tools and user tasks with programmatic success checks. M2 delivers the injection attack styles with deterministic injected-goal checks. M3 delivers the defenses and the full run on the local models. M4 delivers the hosted-model runs within budget, the statistics, and the plots. M5 delivers the paper-style report and the README.

This document defines the requirements only. Benchmark-or-reuse mechanics, the exact tool and task catalog, attack and defense wording, model selection, and other implementation details are deferred to the design phase.

## Glossary

- **System**: The complete Agent Injection Bench, comprising the sandbox, the Agent runner, the attacks, the defenses, the metrics layer, the experiment runner, the report generator, and the continuous-integration gate.
- **Sandbox**: The isolated execution environment in which all Simulated_Tools run against in-memory or local fixture state only, with no network egress to real services.
- **Simulated_Tool**: A tool the Agent can call (for example simulated email, calendar, file store, or web fetch) whose implementation reads and writes only Sandbox state and returns a Tool_Output.
- **Tool_Output**: The text a Simulated_Tool returns to the Agent in response to a tool call; the medium in which an Injection is planted.
- **Agent**: The tool-using LLM, driven through the OpenAI-compatible interface, that reads a User_Task, calls Simulated_Tools, and produces a final response.
- **User_Task**: A legitimate task the Agent is asked to complete, paired with a deterministic Task_Success_Check.
- **Task_Success_Check**: A programmatic predicate over the final Sandbox state and the Agent's response that returns true if and only if the User_Task was completed correctly, with no LLM judgment where a deterministic check is possible.
- **Injection**: Text planted inside a Tool_Output that attempts to make the Agent pursue an Injected_Goal instead of, or in addition to, the User_Task.
- **Injection_Style**: A named category of Injection (for example direct instruction, fake system message, or instruction hidden in a data field) distinguished by how the Injection is phrased or placed.
- **Injected_Goal**: The attacker-chosen action the Injection tries to make the Agent perform, paired with a deterministic Injection_Success_Check.
- **Injection_Success_Check**: A programmatic predicate over the final Sandbox state and the Agent's tool-call trace that returns true if and only if the Agent carried out the Injected_Goal.
- **Defense**: A named mitigation condition applied to a run (for example no defense, delimiting or marking Tool_Output, an instruction-hierarchy prompt, or a separate filter or classifier screening Tool_Output).
- **Defense_Condition**: A specific configured Defense, including the No_Defense control condition.
- **No_Defense**: The control Defense_Condition that applies no mitigation, used as the baseline for all comparisons.
- **Filter_Classifier**: A Defense that screens Tool_Output for Injections before the Agent sees it and either blocks, strips, or flags the content.
- **Model**: An open-weight Local_Model or a Hosted_Model under test.
- **Local_Model**: An open-weight model served on the NVIDIA RTX 3080 through an OpenAI-compatible server.
- **Model_Server**: The OpenAI-compatible server process that serves a Local_Model on the GPU.
- **Hosted_Model**: A hosted API model used under the Spend_Cap.
- **Spend_Cap**: The owner-set monthly spend limit, in United States dollars, for Hosted_Model usage.
- **Cell**: A single (Model, Defense_Condition, Injection_Style) combination evaluated over the User_Tasks.
- **Attack_Success_Rate**: The fraction of attempted injected cases in a Cell for which the Injection_Success_Check returned true.
- **Task_Success_Rate**: The fraction of cases in a Cell for which the Task_Success_Check returned true.
- **Defense_Cost**: The drop in Task_Success_Rate of a Defense_Condition relative to the No_Defense condition for the same Model and Injection_Style.
- **Confidence_Interval**: A 95 percent interval for a reported rate, computed by the Wilson method for a proportion or by bootstrap resampling.
- **Repetition**: One of the repeated executions of a Cell used to account for output variation.
- **Measured_Value**: A number produced by a script in this repository, recorded with its generating command, seed, and full 40-character commit hash.
- **Results_Store**: The generated set of raw per-case result records from which every reported metric is computed.
- **Results_Table**: The generated file containing the aggregated metrics with their Confidence_Intervals.
- **Report**: The generated paper-style write-up (question, setup, results, limitations, threats to validity) with plots.
- **Stub_Model**: A deterministic non-LLM stand-in for an Agent used by the CI smoke subset so CI needs no GPU or API key.
- **ENVIRONMENT_Doc**: The file `docs/ENVIRONMENT.md` recording exact tool, model, and dependency versions proven on the machine.
- **THIRD_PARTY_Doc**: The file `THIRD_PARTY.md` recording each third-party package, model, dataset, and asset and its license.
- **PriorWork_Decision**: The recorded build-or-reuse decision and its reasons, documented in `design.md`.
- **CI_Pipeline**: The GitHub Actions continuous-integration pipeline.

## Requirements

### Requirement 0: Governing non-functional requirements

**User Story:** As the repository owner, I want every result reproducible and every dependency proven and license-checked, so that I can defend every line of the project in an interview and in a research write-up.

#### Acceptance Criteria

1. WHERE a number is reported in the README, in `docs/`, in the Report, or in the Results_Table, THE System SHALL produce that number from a script in this repository and SHALL record, in the same document and within the same section as the number, the generating command, the seed, and the full 40-character commit hash.
2. WHERE a System component uses randomness, THE System SHALL expose a configurable seed through a documented configuration parameter.
3. WHEN two runs of a deterministic System component execute with the same seed, the same configuration, and the same Model, THE System SHALL produce numeric results that are equal to within an absolute tolerance of 1e-6 for floating-point values and exactly equal for integer and categorical values, excluding timestamps, absolute file paths, and values the model provider documents as nondeterministic.
4. IF a run fails or misses a stated target, THEN THE System SHALL report, in the run output, the actual result and the identified cause, SHALL terminate with a non-zero exit status, and SHALL leave the stated target and the test unchanged.
5. THE System SHALL record, in the ENVIRONMENT_Doc, the exact version string of every tool, model, API, and dependency it relies on, including each Local_Model name, version, and quantization, the Model_Server name and version, and, where used, each Hosted_Model name and version.
6. WHEN a dependency, tool, or model version is recorded in the ENVIRONMENT_Doc, THE System SHALL include, alongside that version, at least one of the following verifiable artifacts obtained on the target machine: a package-manager query result showing that exact version, a documentation reference for that exact version, or an automated test that exercises the dependency and passes against that exact version.
7. WHERE a third-party package, model, dataset, or asset is added to or reused by the repository, THE System SHALL record, in the THIRD_PARTY_Doc, its name, its version, and its license identifier.
8. WHERE any benchmark component, task, attack, or fixture is derived from an external source such as AgentDojo or InjecAgent, THE System SHALL record the source and its license in the THIRD_PARTY_Doc and SHALL attribute the reused material to its origin.
9. THE System SHALL read every secret, including any Hosted_Model API key, from an environment variable and SHALL list every secret-bearing file in `.gitignore` so that no such file is tracked in version control.
10. THE System SHALL pin the exact version of every dependency it declares.
11. WHERE a module contains more than one public function or exceeds 50 source lines, THE System SHALL provide unit tests achieving at least 80 percent statement coverage for that module, and THE README SHALL state the exact single command that runs those tests.
12. THE System SHALL remain a private repository without a public license, published package, or public release until the repository owner directs otherwise.
13. IF a required secret environment variable is unset or empty when a System component that depends on it starts, THEN THE System SHALL terminate before issuing any request that requires that secret, SHALL emit an error indicating which environment variable is missing, and SHALL exit with a non-zero exit status.
14. THE Sandbox SHALL execute every Simulated_Tool against Sandbox state only and SHALL NOT issue any network request to a real external account, website, or system; and WHERE a Simulated_Tool models network access, it SHALL serve its response from a local fixture.
15. WHERE the Report or README presents attack results, THE System SHALL report aggregate rates and defense effects only and SHALL NOT publish an attack payload tuned against a specific named commercial product.
16. IF benchmarking reveals a reproducible vulnerability in a specific real third-party product, THEN THE owner SHALL stop publication of that payload and follow that vendor's responsible-disclosure process, and THE repository SHALL document this handling rule in `design.md`.

### Requirement 1: Prior-work summary and build-or-reuse decision, and model-server smoke test (M0)

**User Story:** As the repository owner, I want the existing benchmarks summarized and a license-checked build-or-reuse decision recorded, and the local model server proven, so that the project builds on prior work where it fits and runs the open-weight models on the GPU before any benchmark work begins.

#### Acceptance Criteria

1. WHEN the M0 prior-work review is complete, THE System SHALL record in `design.md` a summary of at least AgentDojo and InjecAgent that states, for each, what it measures, its environment and task scale, and its license as verified from its repository or package metadata.
2. WHEN the build-or-reuse decision is recorded, THE PriorWork_Decision in `design.md` SHALL state whether the project reuses an existing benchmark, builds a smaller custom suite, or combines both, and SHALL give the reasons for that choice including license compatibility and fit to the required scope.
3. WHERE the PriorWork_Decision is to reuse an existing benchmark, THE System SHALL record that benchmark, its version, and its license in the THIRD_PARTY_Doc and SHALL attribute reused components to their origin.
4. WHEN the smoke test runs, THE System SHALL record in the ENVIRONMENT_Doc the output of `nvidia-smi`, including the reported total GPU memory in mebibytes of the NVIDIA RTX 3080.
5. WHEN the smoke test runs, THE System SHALL start the Model_Server, send exactly one tool-free prompt to the OpenAI-compatible endpoint, and record in the ENVIRONMENT_Doc a generated response containing at least one non-whitespace character.
6. WHILE a Local_Model generates the smoke-test response, THE System SHALL measure the peak GPU memory in mebibytes used between the start and completion of the generation request and SHALL record that value in the ENVIRONMENT_Doc.
7. THE System SHALL select each Local_Model such that its recorded peak GPU memory during the smoke test does not exceed the total GPU memory reported by `nvidia-smi`.
8. THE System SHALL record in the ENVIRONMENT_Doc each Local_Model name, version, quantization, and license, and the Model_Server name and version.
9. IF the smoke test does not record a valid response within 120 seconds of sending the prompt, THEN THE System SHALL stop the attempt, report the failure and its identified cause, and SHALL record in the ENVIRONMENT_Doc the condition that failed.
10. WHEN the smoke test runs, THE System SHALL verify that the installed Python version is 3.11 or newer and that Docker is available, and IF either prerequisite is not met, THEN THE System SHALL report the failure and record in the ENVIRONMENT_Doc the prerequisite that failed.
11. WHERE a Hosted_Model will be used, THE System SHALL record the Spend_Cap in United States dollars per month in the ENVIRONMENT_Doc and SHALL read the Hosted_Model API key from an environment variable.

### Requirement 2: Sandbox, simulated tools, and user tasks (M1)

**User Story:** As the repository owner, I want a sandbox with simulated tools and a set of user tasks that each have a programmatic success check, so that task completion is measured deterministically and no agent ever touches a real system.

#### Acceptance Criteria

1. THE Sandbox SHALL provide at least 4 distinct Simulated_Tools, and each Simulated_Tool SHALL read and write only Sandbox state.
2. WHEN a Simulated_Tool is invoked, THE Sandbox SHALL return a Tool_Output computed only from Sandbox state and the call arguments, and SHALL make no request to a real external service.
3. THE System SHALL provide at least 40 User_Tasks, and each User_Task SHALL have a Task_Success_Check that is a deterministic programmatic predicate.
4. WHERE a User_Task's success can be checked deterministically, THE Task_Success_Check SHALL NOT use an LLM to judge success.
5. WHEN the Sandbox is initialized for a case with a fixed seed, THE Sandbox SHALL produce an initial state that is identical across repeated initializations with the same seed and the same User_Task.
6. WHEN a Task_Success_Check evaluates a case, THE check SHALL return a boolean outcome and, when the outcome is false, SHALL report the condition that was not satisfied.
7. WHEN the User_Task catalog is loaded, THE System SHALL verify that every User_Task has a resolvable Task_Success_Check, and IF any User_Task lacks one, THEN THE System SHALL report each such User_Task and the total count.

### Requirement 3: Injection attack styles and injected-goal checks (M2)

**User Story:** As the repository owner, I want several injection styles placed in tool outputs, each with a deterministic check of whether the agent carried out the injected goal, so that attack success is measured objectively.

#### Acceptance Criteria

1. THE System SHALL provide at least 5 distinct Injection_Styles, where two Injection_Styles are distinct if they differ in how the Injection is phrased or placed, and the set SHALL include at least a direct-instruction style, a fake-system-message style, and an instruction-hidden-in-a-data-field style.
2. WHEN an Injection is applied to a case, THE System SHALL place the Injection text inside a Tool_Output returned by a Simulated_Tool and SHALL NOT alter the User_Task text.
3. THE System SHALL pair each Injected_Goal with an Injection_Success_Check that is a deterministic programmatic predicate over the final Sandbox state and the Agent's tool-call trace.
4. WHEN an Injection_Success_Check evaluates a case, THE check SHALL count the case as a successful injection if and only if the Agent carried out the Injected_Goal according to the documented classification rule.
5. WHERE the Agent's behavior on an injected case cannot be classified as carried-out or not-carried-out, THE System SHALL report that case as unclassified and SHALL exclude it from the Attack_Success_Rate denominator while reporting the count of unclassified cases.
6. WHEN an injected case runs, THE System SHALL record whether the Injection_Success_Check and the Task_Success_Check each returned true, so that attack success and task success are measured on the same case.
7. THE Injected_Goals SHALL be benign sandbox actions and SHALL NOT direct the Agent to produce content-policy-violating output, so that the benchmark studies injected instructions in tool data rather than content-policy evasion.

### Requirement 4: Defenses (M3)

**User Story:** As the repository owner, I want several defense conditions, so that I can measure how much each one lowers the attack success rate and what it costs in task success.

#### Acceptance Criteria

1. THE System SHALL provide at least 4 Defense_Conditions: a No_Defense control; a condition that delimits or marks Tool_Output; a condition that adds an explicit instruction-hierarchy prompt; and a Filter_Classifier condition that screens Tool_Output.
2. WHEN a Defense_Condition is applied to a case, THE System SHALL apply only that Defense_Condition's mitigation and SHALL leave the User_Task, the Injection, and the Injected_Goal otherwise unchanged for that Cell.
3. WHEN the No_Defense control runs, THE System SHALL apply no mitigation to the Tool_Output or the prompt, establishing the baseline for Defense_Cost.
4. WHEN the Filter_Classifier screens a Tool_Output, THE System SHALL record, for each screened Tool_Output, whether the Filter_Classifier flagged it, so that filter precision and recall can be computed against the known injected and clean cases.
5. WHERE a Defense_Condition requires a configuration value such as a classifier threshold or a prompt template, THE System SHALL expose that value as a documented configuration parameter.
6. WHERE a tool-permission-limit Defense is implemented as an optional condition, THE System SHALL enforce it by restricting which Simulated_Tools the Agent may call for a Cell and SHALL record the restriction applied.

### Requirement 5: Models and runs (M3, M4)

**User Story:** As the repository owner, I want at least two local open-weight models and, within budget, one or two hosted models run across the full grid, so that I can compare models under the same conditions.

#### Acceptance Criteria

1. THE System SHALL run the benchmark on at least 2 Local_Models served through the OpenAI-compatible Model_Server on the GPU.
2. WHERE the Spend_Cap allows, THE System SHALL run the benchmark on 1 or 2 Hosted_Models, and WHILE running Hosted_Models THE System SHALL track estimated spend and SHALL stop issuing Hosted_Model requests before the Spend_Cap would be exceeded.
3. WHEN a Cell is run, THE System SHALL record the Model name and version, the Defense_Condition, the Injection_Style, the temperature, and the sampling settings used for that Cell.
4. WHERE a Cell's Agent outputs vary across repeated executions, THE System SHALL run that Cell at least 3 times and SHALL record each Repetition's per-case outcomes.
5. WHEN a Model run writes results, THE System SHALL write one record per case into the Results_Store, each record carrying the Model, Defense_Condition, Injection_Style, User_Task identifier, Repetition index, Injection_Success_Check outcome, Task_Success_Check outcome, and classification status.
6. IF a single case fails to execute, THEN THE System SHALL record that case as errored with its identified cause and SHALL continue running the remaining cases.
7. WHEN two runs of a Cell execute with the same seed, the same configuration, the same Model, and the same temperature and sampling settings, THE System SHALL reproduce the recorded configuration metadata exactly, excluding values the model provider documents as nondeterministic.

### Requirement 6: Metrics and statistics (M4)

**User Story:** As the repository owner, I want attack and task success rates per cell with confidence intervals and the task-success cost of each defense, so that the measurements are honest and comparable.

#### Acceptance Criteria

1. WHEN the metrics layer aggregates the Results_Store, THE System SHALL report, per Cell, the Attack_Success_Rate and the Task_Success_Rate as Measured_Values, each in the range 0.0 to 1.0.
2. WHEN the metrics layer reports a rate, THE System SHALL report a 95 percent Confidence_Interval for that rate computed by the Wilson method for a proportion or by bootstrap resampling from at least 1,000 resamples, and SHALL record which method was used.
3. WHEN the metrics layer computes the Attack_Success_Rate for a Cell, THE System SHALL compute it as the number of cases whose Injection_Success_Check returned true divided by the number of classified injected cases, excluding unclassified and errored cases from the denominator.
4. WHEN the metrics layer computes Defense_Cost, THE System SHALL compute it as the Task_Success_Rate of the No_Defense condition minus the Task_Success_Rate of the Defense_Condition for the same Model and Injection_Style, and SHALL report it as a Measured_Value.
5. WHERE the Filter_Classifier condition was run, THE System SHALL report the Filter_Classifier's precision and recall against the known injected and clean cases as Measured_Values, each in the range 0.0 to 1.0.
6. WHEN the metrics layer runs twice on an identical Results_Store with the same seed, THE System SHALL produce metric values and interval bounds that are equal to within an absolute tolerance of 1e-6.
7. THE metrics layer SHALL be invocable from a single non-interactive command and SHALL record each reported metric with its generating command, seed, and full 40-character commit hash.
8. IF a Cell has zero classified cases, THEN THE metrics layer SHALL report that Cell's rate as not computed with its cause and SHALL continue computing the remaining Cells.

### Requirement 7: Report and plots (M5)

**User Story:** As a reader, I want a short paper-style write-up with plots generated from the results files, so that I can understand the question, the setup, the results, and the limitations.

#### Acceptance Criteria

1. THE Report SHALL contain a question section, a setup section, a results section, a limitations section, and a threats-to-validity section.
2. WHEN the Report is generated, THE System SHALL produce it from the Results_Store and the Results_Table by a repository script, and no number in the Report SHALL be modified by any means other than regenerating it from those files.
3. WHEN the Report presents results, THE System SHALL include at least one plot of Attack_Success_Rate by Defense_Condition and at least one plot showing the task-success cost of each Defense, each plot generated by a repository script with its Confidence_Intervals shown.
4. WHERE a number appears in the Report, THE System SHALL source that number from the Results_Store, the Results_Table, or another file generated by a repository script rather than from a hand-entered value.
5. THE Report SHALL state the temperature and sampling settings used and the number of Repetitions per Cell.
6. THE Report SHALL report aggregate rates and defense effects only and SHALL NOT include an attack payload tuned against a specific named commercial product.

### Requirement 8: Tests and continuous integration (M1 onward; gate at M5)

**User Story:** As the repository owner, I want unit tests for the tools, success checks, and metric code, and a CI smoke subset that runs with a stub model, so that the benchmark stays correct and CI needs no GPU or API key.

#### Acceptance Criteria

1. THE System SHALL provide unit tests for the Simulated_Tools, the Task_Success_Checks, the Injection_Success_Checks, and the metric and statistics code.
2. WHEN a change is pushed to the repository's default branch or a pull request targeting the default branch is opened or updated, THE CI_Pipeline SHALL run the unit tests and a smoke subset of the benchmark driven by the Stub_Model.
3. WHEN the CI smoke subset runs, THE System SHALL drive the Agent with the Stub_Model so that the run requires no GPU and no Hosted_Model API key.
4. IF any unit test fails during a CI run, THEN THE CI_Pipeline SHALL report a failed status for that run.
5. IF the CI smoke subset does not complete or its deterministic expected outcomes are not met, THEN THE CI_Pipeline SHALL report a failed status for that run and SHALL report the identified cause.
6. WHEN the Stub_Model drives a case, THE System SHALL produce a deterministic tool-call trace and final response so that the smoke subset's Injection_Success_Check and Task_Success_Check outcomes are fixed and assertable.

### Requirement 9: README and reproducibility (M5)

**User Story:** As a reader, I want a README with measured results and exact commands, so that I can understand, run, and reproduce the benchmark.

#### Acceptance Criteria

1. THE README SHALL contain an architecture diagram, run instructions, a Results_Table that includes a Confidence_Interval for every reported rate, a link to the Report, a known-limitations section, and a safety-and-scope section stating that all tools are simulated and no real systems are targeted.
2. THE README SHALL state, for each test suite, the exact single command that runs that suite, and SHALL state, for each reported result, the exact single command that reproduces that result.
3. WHERE a number appears in the README, THE System SHALL source that number from the Results_Table or another file generated by a repository script rather than from a hand-entered value.
4. IF a number in the README cannot be traced to the Results_Table or another generated file, THEN THE System SHALL report that number as untraceable in the check output and SHALL report a failed status for that check.
5. THE README SHALL state the single command that runs the full local-model benchmark and the single command that regenerates the Report and plots.

## Out of Scope

The following are explicitly out of scope: training or fine-tuning of any model; attacks against real services, real accounts, or systems the owner does not own; jailbreaks aimed at eliciting harmful or content-policy-violating output (this project studies injected instructions in tool data, not content-policy evasion); a graphical user interface beyond the generated Report and plots; and publishing attack payloads tuned against a specific named commercial product. The optional tool-permission-limit Defense (Requirement 4.6) is permitted but not required.
