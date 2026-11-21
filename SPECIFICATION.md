# System Specification: Grounded Multimodal AI Tutor

**Document Type:** Product & Functional Specification (Spec-Driven Development)  
**Status:** Approved for Implementation Planning  
**Target Course Domain:** Higher Education STEM Instruction (Bounded Prototype: MIT 6.006 *Introduction to Algorithms*)

---

## 1. Executive Summary & Problem Statement

### 1.1 Context
In university-level educational environments, a learning assistant must instruct strictly from approved course material, demonstrate the exact empirical evidence underlying every assertion, and decisively avoid confidently fabricating unsupported claims. Generic conversational artificial intelligence models, when applied to educational tasks, routinely produce plausible-sounding hallucinations, invent mathematical or algorithmic properties, and lack accountability.

Furthermore, university lectures in technical disciplines are inherently multimodal. Key explanations, data structures, recursion trees, state machines, and code snippets are presented on blackboards and slide decks. Instructors frequently rely on deictic gestures and verbal indexicals (*"as shown here"*, *"this branch terminates"*, *"following this path"*) without speaking the complete visual content into the audio stream. A text-only or transcript-only assistant is functionally blind to these instructional moments.

Finally, effective pedagogical personalization requires understanding a student's evolving cognitive state, misconceptions, and goals over time. However, naively injecting raw dialogue histories into language model contexts pollutes the reasoning buffer, leaks prompt injections, and risks permanently cementing hallucinated student attributes.

### 1.2 Problem Statement
To design, specify, and evaluate a grounded multimodal AI tutor that:
1. Performs hybrid and cross-modal retrieval over authentic public course materials (video, audio, slides, notes).
2. Generates strictly grounded explanations featuring verifiable source and timestamp citations precise enough for instant navigation to the exact supporting moment or visual artifact.
3. Solves a dedicated benchmark subset of instructional questions that are strictly impossible to answer from spoken transcripts alone.
4. Maintains a policy-controlled, auditable learner memory separating ephemeral dialogue from durable profiles, governed by student confirmation and complete user CRUD control.
5. Adheres to strict abstention and source-version safety policies by refusing to guess on unsupported topics and invalidating superseded errata.
6. Provides an interactive minimal frontend interface for student learning, evidence navigation, and profile governance.
7. Incorporates an automated adversarial evaluation harness quantifying retrieval quality, answer correctness, citation validity, abstention accuracy, stale-content handling, and injection resistance across empirical configurations.

---

## 2. Project Vision & Core Principles (What and Why)

### 2.1 What Is the System?
The Grounded Multimodal AI Tutor is a verifiable educational copilot that reads, listens to, and observes course instructional materials to guide students through technical subjects. It pairs natural-language explanations with exact, navigable references to the video moments and slide images where the concept was taught, while offering the student full transparency and control over what the assistant remembers about their learning journey.

### 2.2 Why Are We Building It?
* **Eliminate Academic Hallucinations**: In STEM education, inaccurate information directly misleads students. The tutor must provide proof for its claims or abstain.
* **Bridge the Spoken–Visual Gap**: Instructional blackboard diagrams and visual slide layouts contain essential knowledge omitted from spoken transcripts.
* **Protect Student Agency & Cognitive Safety**: Learners must own their profile data. The system must never silently track or assume permanent student deficits without verification.
* **Ensure Content Currency**: Academic curricula evolve. The tutor must recognize when source material is updated and prevent stale or retracted notes from being served as truth.

### 2.3 Core Operating Principles
1. **Evidence-Bound Instruction**: No factual assertion may be presented without an underlying evidence binding.
2. **Tri-State Epistemic Humility**: The tutor operates in three distinct states:
   * *Full Grounded Answer*: Strong multimodal/textual evidence exists $\rightarrow$ answer with direct citations and visual assets.
   * *Qualified Answer*: Partial evidence exists $\rightarrow$ answer cautiously, state what cannot be proven, and direct the student to inspect the source material.
   * *Explicit Abstention*: No approved evidence exists or question is off-topic $\rightarrow$ refuse to speculate and state the boundary of knowledge clearly.
3. **Zero-Trust Memory Separation**: Ephemeral conversational turns must never automatically become permanent cognitive traits. Durable memory requires policy gating and student approval.
4. **Verifiable Auditability**: Every citation must be programmatically checked for validity, bounds accuracy, and file accessibility before it is displayed to the learner.

---

## 3. User Personas & User Journeys

### 3.1 Personas
* **The Student (Learner)**: Seeks conceptual clarification, algorithmic analysis, and problem-solving help. Needs trustworthy explanations, visual clarity on complex diagrams, direct access to the lecture video moment, and the ability to view/edit what the tutor remembers about them.
* **The Evaluator / Red Teamer**: Tests the system against adversarial prompt injections, off-corpus queries, visual-only questions, and superseded content to measure factual grounding, abstention precision, and safety.
* **The Course Instructor / Administrator**: Updates course material, issues errata, invalidates outdated lecture revisions, and inspects tutor accuracy benchmarks.

### 3.2 Key User Journeys
1. **Asking a Visually Grounded Question**:
   * *Action*: Student asks about a diagram shown during a lecture (e.g., how a recursion tree branches or what variables are on a slide).
   * *Result*: Tutor explains the concept, cites the exact video timestamp interval, displays the corresponding high-resolution slide or video frame, and allows the student to jump directly to that point in the lecture.
2. **Experiencing Safe Abstention**:
   * *Action*: Student asks a question outside the syllabus (e.g., world history or an advanced topic not yet covered).
   * *Result*: Tutor politely declines, states that the topic is not covered in the approved course material, and avoids fabricating an answer.
3. **Governing Learner Memory**:
   * *Action*: During discussion, the tutor identifies that the student is confusing two sorting approaches and proposes saving a learning note.
   * *Result*: The student is prompted to confirm, edit, or reject the note. In the dedicated memory management view, the student can inspect all active memory records and delete any item at will.
4. **Encountering Superseded Content**:
   * *Action*: Student asks about a concept where an erratum or lecture revision was published.
   * *Result*: Tutor alerts the student that the initial explanation was superseded by an update, providing the current corrected version.

---

## 4. Scope and System Boundaries

### 4.1 In-Scope
* **Bounded Public Instructional Material**: Initial deployment bounded to 2–3 named lectures (~2 hours) of university-level instruction (e.g., MIT OCW 6.006 *Introduction to Algorithms*).
* **Multi-Modal Data Types**: Synchronized audio transcripts, 1080p video keyframes, vector PDF slide decks, and code/algorithmic listings.
* **Interactive Minimal Frontend**: A functional, accessible user interface combining a chat workspace, an interactive evidence/video player, and a learner memory governance panel.
* **Automated Evaluation Harness**: Comprehensive regression suite executing a 20-query benchmark covering text retrieval, visual necessity, reasoning, abstention, staleness, and prompt injection.

### 4.2 Out-of-Scope
* Autonomous grading or formal exam administration.
* Full-scale LMS integration (Canvas, Blackboard) or enterprise user identity federation.
* Real-time conversational speech-to-speech audio streaming.
* Proprietary model pre-training or fine-tuning (the system relies on frozen, instruction-following reasoning backbones).

---

## 5. Milestone Breakdown (The 8 Milestones)

### M1 – Versioned Evidence Corpus
* **Requirement**: Ingest and structure public instructional materials across audio transcripts, video frames, slide images, and reading text into an interconnected knowledge corpus.
* **Capabilities**:
  * Every indexed chunk must possess an immutable identity carrying its canonical source identifier, content hash (cryptographic digest), semantic content version, and precise temporal boundaries (`start_ms`, `end_ms`).
  * Cross-reference links must bind spoken sentences directly to the visual frame and slide page visible at that exact second.
* **Acceptance Criteria**: Any indexed evidence chunk can be programmatically traced back to its raw physical asset, content hash, and timestamp offset.

### M2 – Hybrid Retrieval & Fusion
* **Requirement**: Provide a multi-channel retrieval engine capable of matching queries across distinct retrieval modes and query classes.
* **Capabilities**:
  * *Lexical/Sparse Channel*: Captures exact mathematical symbols, variable identifiers, and asymptotic notation (e.g., $O(n \log n)$).
  * *Semantic/Dense Channel*: Captures natural conceptual phrasing and intent.
  * *Visual/Cross-Modal Channel*: Matches textual user queries directly to diagrams, charts, and slide images.
  * *Rank Fusion & Reranking*: Merges candidate sets across all channels via rank-based fusion and supports an optional second-stage reranker for ambiguous or high-recall queries.
* **Acceptance Criteria**: The retrieval pipeline places the true answer-bearing chunk in the top candidate set for both exact-syntax queries and broad conceptual questions.

### M3 – Timestamp-Grounded Answers & Citation Contract
* **Requirement**: Every generated response must be bound to verified evidence via precise, machine-readable citations.
* **Capabilities**:
  * Citations follow a strict contract linking each factual claim to one or more evidence identifiers.
  * Server-side pre-display validation verifies that cited evidence exists, is currently active, has valid timestamp bounds, and references an existing image asset.
  * Invalid or hallucinated citation keys are programmatically intercepted and flagged before presentation.
* **Acceptance Criteria**: 100% of generated citations resolve to active, playable lecture intervals or accessible visual assets. Zero fabricated citation identifiers reach the user.

### M4 – Multimodal Evidence & Visual Benchmark
* **Requirement**: The tutor must reliably resolve instructional queries that cannot be answered from spoken transcripts alone.
* **Capabilities**:
  * Extract structural, geometric, and textual meaning from visual artifacts (recursion tree tiers, highlighted blackboard matrix arrows, variable names in slide code).
  * Serve the exact visual asset (slide or keyframe) alongside the written answer.
* **Acceptance Criteria**: On a dedicated benchmark subset of at least 5 visually required queries where transcript-only baselines fail, the multimodal tutor achieves a measurable accuracy superiority ($\Delta_{\text{visual}} \ge 0.40$).

### M5 – Selective Learner Memory & Governance
* **Requirement**: Provide a policy-controlled learner memory architecture that isolates ephemeral chat turns from durable student profiles.
* **Capabilities**:
  * *Ephemeral Tier*: Short-term session buffer managing conversational flow with token-pressure auto-summarization.
  * *Durable Tier*: Structured records categorized strictly by `goal`, `mastery`, `misconception`, or `preference`.
  * *Confirmation Gate*: The tutor proposes durable notes based on pedagogical signals; notes are persisted only upon student verification.
  * *Lifecycle Policies*: Configurable retention, confidence decay over time ($t_{1/2} = 30$ days), and TTL expiration dates.
  * *User CRUD*: Full user capabilities to Inspect, Correct, and Permanently Delete memory records.
* **Acceptance Criteria**: A deleted memory record immediately ceases to appear in or influence any subsequent tutor prompt context.

### M6 – Abstention & Version Safety
* **Requirement**: Enforce epistemic boundaries and protect against stale, superseded instructional data.
* **Capabilities**:
  * Multi-signal decision engine evaluating retrieval scores, query coverage, and on-corpus classification to trigger full answers, qualified answers, or explicit refusals.
  * Cascading invalidation: Updating a course asset marks prior versions as `superseded`, purging dependent cached summaries and triggering warnings if an older source is queried.
* **Acceptance Criteria**: 100% of out-of-syllabus and unsupported questions produce safe abstentions or qualified hedges; zero confident hallucinations occur off-corpus.

### M7 – Automated RAG Evaluation & Red Team
* **Requirement**: Provide a fully automated, reproducible evaluation harness measuring retrieval and generation quality alongside adversarial security.
* **Capabilities**:
  * Quantitative scoring covering six dimensions: Retrieval Precision/Recall, Answer Faithfulness, Citation Precision, Abstention F1, Visual Necessity Delta, and Prompt Injection Resistance Rate.
  * Standardized evaluation fixture of 20 labelled queries including positive controls, visual-required cases, off-corpus questions, stale-source fixtures, and direct/indirect prompt injection attacks.
* **Acceptance Criteria**: The evaluation harness executes automatically and outputs reproducible quantitative metrics across all six dimensions.

### M8 – Performance & Tradeoff Characterization
* **Requirement**: Characterize and publish the empirical trade-offs between response quality and latency across distinct pipeline configurations.
* **Capabilities**:
  * Profile at least two distinct operational configurations:
    * *Configuration A (Quality-Optimized)*: Multi-channel retrieval + cross-encoder reranking.
    * *Configuration B (Speed-Optimized)*: Multi-channel retrieval skipping reranking with shallow prefetch depth.
  * Measure end-to-end latency percentiles (p50 and p95), stage-by-stage breakdowns (retrieval vs. rerank vs. generation), and answer quality.
* **Acceptance Criteria**: A Pareto frontier comparison report documenting the trade-off curve between response latency and answer precision.

---

## 6. Minimal Frontend / User Interface Requirements

To ensure the tutor is accessible to evaluators and learners rather than remaining an isolated backend, the system must incorporate a cohesive, minimal frontend interface.

```
+---------------------------------------------------------------------------------------+
|                              GROUNDED MULTIMODAL AI TUTOR                             |
+---------------------------------------------------+-----------------------------------+
|                  CHAT WORKSPACE                   |          EVIDENCE VIEWER          |
|                                                   |                                   |
| [Tutor]: Welcome! What concept would you like     |  [Visual Slide / Keyframe Panel]  |
| to explore today?                                 |  +-----------------------------+  |
|                                                   |  |                             |  |
| [Student]: In merge sort, how much work is done   |  |   Recursion Tree Diagram    |  |
| at level 2 of the recursion tree?                 |  |   Level 0: cn               |  |
|                                                   |  |   Level 1: cn/2 + cn/2 = cn |  |
| [Tutor]: Across all 4 subproblems at level 2 of   |  |   Level 2: 4 * (cn/4) = cn  |  |
| the recursion tree, the total work is c * n.      |  |                             |  |
| Each subproblem takes c * (n / 4) work.           |  +-----------------------------+  |
| 📎 [Lecture 3 @ 14:32]   📊 [Slide 12]            |  Source: 6.006 Lecture 3, Slide 12|
|                                                   |                                   |
| 💡 [Tutor Proposal]: You seem interested in       |  [Synchronized Video Navigator]   |
| recurrence analysis. Should I remember:           |  Jump to: [ 14:32 ]               |
| "Mastering recursion trees for sorting"?          |  Duration: 14:32 - 15:01          |
| [ Confirm ]  [ Edit ]  [ Dismiss ]                |  Status: Active (Corpus v1.0)     |
|                                                   |                                   |
|                                                   +-----------------------------------+
|                                                   |      LEARNER PROFILE PANEL        |
|                                                   |                                   |
|                                                   | - [GOAL] Prepare for Midterm [X]  |
|                                                   | - [MASTERY] Peak Finding (High)[X]|
|                                                   | - [MISCONCEPTION] None active     |
|                                                   |                                   |
|                                                   | [ View Full Profile / Audit Log ] |
+---------------------------------------------------+-----------------------------------+
| Input: [ Type your algorithmic question here...                             ] [Send]  |
+---------------------------------------------------------------------------------------+
```

### 6.1 View 1: Interactive Chat Workspace
* **Conversational Stream**: Displays turn-by-turn dialogue between the learner and the tutor.
* **Interactive Citation Badges**: Renders verified citations as clickable badges (e.g., `📎 [Lecture 3 @ 14:32]` or `📊 [Slide 12]`). Clicking a badge updates the synchronized evidence viewer immediately.
* **Epistemic Qualification Banners**: Clearly distinguishes between full answers, qualified answers (hedging banner with disclaimer), and explicit abstention notices.
* **In-Stream Memory Proposal Cards**: Displays the Confirmation Gate when the tutor infers a learning insight, providing one-click "Confirm", "Edit", and "Dismiss" actions.

### 6.2 View 2: Synchronized Evidence & Media Viewer
* **Image Asset Pane**: Renders high-resolution keyframes and rendered slide images cited in the active answer. Supports click-to-expand modal inspection for detailed mathematical diagrams.
* **Temporal Navigation Controller**: Displays video source metadata, start/end time ranges, and provides a direct timestamp jump controller to launch or seek the lecture video to the cited second.
* **Corpus & Version Tag**: Explicitly displays the source identifier, content version, and active status of the rendered evidence. Emits a visual warning badge if evidence is marked superseded.

### 6.3 View 3: Learner Memory Governance Panel
* **Active Profile List**: Clean, tabular view displaying all currently held durable memory records grouped by category (`Goal`, `Mastery`, `Misconception`, `Preference`).
* **Metadata & Confidence Transparency**: Shows the epistemic confidence score, date created, last reinforced date, and the specific dialogue excerpt (provenance) that triggered each memory.
* **Direct Student Modification (CRUD)**:
  * *Edit*: Allows students to modify the phrasing of a misconception or goal.
  * *Delete*: One-click hard deletion removing the record permanently from the database.
  * *Manual Add*: Allows students to author their own goals or preferences directly.

---

## 7. Functional Requirements (FR)

| ID | Title | Description | Milestone |
|---|---|---|---|
| **FR-01** | Multi-Channel Ingestion | The system shall ingest audio streams, video frames, slide decks, and lecture notes into unified, temporally aligned segments. | M1 |
| **FR-02** | Cryptographic Versioning | Every indexed chunk shall carry a cryptographic hash, a semantic version string, and a lifecycle status flag (`active` or `superseded`). | M1, M6 |
| **FR-03** | Hybrid Multimodal Retrieval | The system shall execute parallel queries across sparse keyword matching, dense conceptual semantics, and cross-modal visual representations. | M2 |
| **FR-04** | Rank Fusion & Reranking | The retrieval subsystem shall combine multi-channel candidate lists using rank fusion and support an optional cross-encoder reranking stage. | M2 |
| **FR-05** | Strict Inline Citations | All factual claims synthesized by the tutor must contain machine-readable citation tokens referencing specific retrieved evidence identifiers. | M3 |
| **FR-06** | Pre-Display Validation Gate | The system shall deterministically verify that all cited evidence exists, is active, has valid timestamp bounds, and has accessible assets before rendering. | M3 |
| **FR-07** | Visually Grounded Reasoning | The tutor shall answer questions requiring visual interpretation of slides, trees, diagrams, and blackboard drawings, returning the supporting image asset. | M4 |
| **FR-08** | Two-Tier Memory Isolation | The system shall isolate ephemeral dialogue buffers from durable learner records, preventing raw conversation text from entering durable storage. | M5 |
| **FR-09** | Memory Confirmation Gate | The tutor shall propose durable memory additions to the learner; no durable record may be persisted without student confirmation or authoring. | M5 |
| **FR-10** | Memory Lifecycle & Decay | Durable memory records shall support exponential confidence decay based on elapsed time and configurable time-to-live (TTL) expiration dates. | M5 |
| **FR-11** | User Memory Governance (CRUD) | The system shall provide student interfaces to inspect, author, modify, and permanently delete any durable memory record. | M5 |
| **FR-12** | Tri-State Abstention Engine | The system shall evaluate retrieval confidence and query relevance to output either a full cited answer, a qualified answer, or an explicit refusal. | M6 |
| **FR-13** | Cascading Version Invalidation | Publishing a new version of course material shall automatically mutate prior version records to `superseded` and invalidate dependent cached summaries. | M6 |
| **FR-14** | Automated Evaluation Harness | The system shall execute an automated benchmark measuring factual faithfulness, citation precision, abstention F1, and visual accuracy delta. | M7 |
| **FR-15** | Adversarial Red Teaming | The evaluation suite shall test resilience against direct prompt injection overrides and indirect memory-poisoning payloads. | M7 |
| **FR-16** | Performance Profiling | The system shall record and report latency distributions and quality scores across at least two distinct operational configurations. | M8 |
| **FR-17** | Interactive Frontend Interface | The system shall provide an integrated web interface featuring chat, synchronized media evidence inspection, and learner memory management. | Frontend |

---

## 8. Non-Functional Requirements (NFR)

### 8.1 Grounding, Verifiability & Accuracy
* **NFR-01 (Zero Hallucinated Citations)**: Exactly $0.0\%$ of citations presented to the student may reference fictitious evidence identifiers or non-existent timestamps.
* **NFR-02 (Citation Precision)**: $\ge 90\%$ of generated factual claims must be strictly entailed by their cited evidence segments.
* **NFR-03 (Visual Accuracy Delta)**: On the dedicated visual-required benchmark subset, the multimodal tutor must achieve an accuracy margin $\Delta_{\text{visual}} \ge 0.40$ over transcript-only baselines.

### 8.2 Safety, Privacy & Security
* **NFR-04 (Abstention Precision)**: On out-of-syllabus and ungrounded queries, the system must achieve an Abstention Precision $\ge 95\%$, admitting lack of evidence rather than fabricating answers.
* **NFR-05 (Prompt Injection Resistance)**: The system must achieve a $100\%$ defense rate against direct persona overrides and zero successful memory poisoning attacks via user input.
* **NFR-06 (Data Privacy & Erasure)**: When a student executes a memory deletion, the target record must be deleted immediately; it must never appear in subsequent context windows.

### 8.3 Latency & Operational Responsiveness
* **NFR-07 (Conversational Latency Bounds)**:
  * *Speed-Optimized Profile (Config B)*: Median response latency $\le 1.5$ seconds (tail p95 $\le 2.5$ seconds).
  * *Quality-Optimized Profile (Config A)*: Median response latency $\le 3.5$ seconds (tail p95 $\le 5.0$ seconds).
* **NFR-08 (Resource Footprint)**: All client-side components and embedded retrieval caches must operate efficiently on standard commodity hardware without requiring enterprise cluster infrastructure.

### 8.4 Usability & Accessibility
* **NFR-09 (Evidence Navigability)**: Clicking any verified citation must jump the synchronized viewer to the exact supporting second within $\le 300$ milliseconds.
* **NFR-10 (Epistemic Transparency)**: The user interface must distinctly color-code and label full answers, qualified hedges, and explicit abstentions so learners are never misled about confidence levels.

---

## 9. Definition of Done (DoD) & Acceptance Sign-Off

The system is considered functionally complete and accepted when an independent evaluator can verify that:
1. **Multimodal Query Execution**: Can submit text-only and visually dependent questions over unseen public course material and receive factually accurate, cited responses accompanied by correct visual assets.
2. **Citation Audit**: Every generated citation correctly links to an active instructional moment and video timestamp interval that precisely supports the claim.
3. **Memory Governance Audit**: Can view the tutor’s persistent profile, edit an entry, delete an entry, and verify via subsequent turns that deleted information is completely purged.
4. **Adversarial & Abstention Verification**: Off-corpus queries reliably trigger explicit refusals; direct jailbreak attempts fail to compromise the tutor persona.
5. **Version Staleness Verification**: When an asset is marked superseded, queries depending on it produce explicit staleness warnings or refresh against the current version.
6. **Benchmark Reproducibility**: Running the automated test harness executes all 20 labelled test queries and outputs reproducible quality, citation, safety, and latency scores across both benchmark configurations.

