# HCLTech Career Shaper Programme
## Ethical Compliance Declaration — Industry Project 73

**Project Title:** LLM-Powered Customer Support Chatbot with Memory  
**Project No.:** 73  
**Domain:** LLM / Python  
**Academic Year:** 2025–26  
**Format:** Individual Project  
**Author / Learner:** [Student Name / Learner ID]  
**Supervisor / Mentor:** Gaur Shaily (`gaur-shaily@hcltech.com`)

---

### Formal Declaration & Statement of Compliance

I hereby declare and affirm that the design, implementation, and deployment of the **LLM-Powered Customer Support Chatbot with Memory** (Project No. 73) rigorously adheres to the ethical principles, compliance guidelines, and safety standards mandated by the **HCLTech Career Shaper Programme** and international AI governance frameworks (including IEEE Ethically Aligned Design and the EU AI Act guidelines for conversational systems).

Specifically, I confirm the implementation of the following core ethical safeguards:

#### 1. Mandatory AI Identity Disclosure
- **Transparency Mandate:** The conversational agent explicitly identifies itself as an Artificial Intelligence system ("Nova") upon every initial greeting and system disclosure banner.
- **Deception Prevention:** The bot never falsely presents itself as a human customer service representative.
- **User Agency:** Users are actively informed that their conversation is processed using Natural Language Processing (NLP) and Retrieval-Augmented Generation (RAG).

#### 2. Prohibition on Sensitive & Personal Decision-Making
- **Scope Restriction:** The chatbot is strictly engineered for enterprise product FAQs, technical navigation, billing assistance, shipping logistics, and customer service routing.
- **Exclusion of High-Risk Domains:** Under no circumstances is the system deployed or authorized to make decisions regarding medical advice, legal judgment, creditworthiness scoring, employment eligibility, or critical personal safety matters.
- **Dispute Escalation:** Financial charge disputes and formal complaints are automatically routed to verified human tier-2 operators with complete conversational audit trails.

#### 3. Content Filtering & Harm Mitigation
- **Input/Output Guardrails:** The system incorporates heuristic and regex-based content guardrails to detect, intercept, and refuse profane, abusive, harassing, or sexually explicit content.
- **Adversarial Robustness:** Prompt injection defenses and strict system-level delimiters prevent users from overriding core safety instructions or exfiltrating underlying system prompts.
- **Hallucination Suppression:** RAG grounding restricts factual claims to indexed FAQ entries (`knowledge_base/faqs.json`). When external ungrounded queries arise, the bot gracefully falls back or suggests human agent contact rather than generating unverified assertions.

#### 4. Data Privacy, PII Protection & Storage Ethics
- **Personally Identifiable Information (PII):** Customer support sessions are indexed solely via pseudonymized alphanumeric identifiers (`session_id`). Raw credit card CVVs, account passwords, and sensitive credentials are never stored in memory logs or vectorized datasets.
- **Memory Retention Controls:** Conversation memory uses sliding context windows (last 6 turns) and time-to-live expiration policies, avoiding indefinite retention of customer chat interactions.
- **Non-Training Guarantee:** Customer queries and dialogue turns are strictly prohibited from being transmitted to third parties or used to train open public foundation models without explicit organizational consent.

#### 5. Human-in-the-Loop (HITL) Fallback & Escalation
- **Guaranteed Escalation Path:** Customers retain uninhibited rights to bypass automated interaction and request human intervention. Typing trigger keywords (`agent`, `supervisor`, `human`) or triggering low confidence thresholds (`< 0.68`) immediately activates human handoff protocol.

---

### Compliance Sign-Off

| Field | Details |
| :--- | :--- |
| **Learner Name** | [Learner / Student Name] |
| **Learner ID / Roll No.** | [Learner Roll Number] |
| **Submission Date** | October 2026 |
| **Declaration Status** | **COMPLIANT & SIGNED** |
| **Digital Signature** | *[Signed Electronically — Learner]* |
| **Mentor Reviewer** | Gaur Shaily (`gaur-shaily@hcltech.com`) |

---
*HCLTech Career Shaper Programme | Academic Year 2025–26 | Individual Project — 4 Weeks*
