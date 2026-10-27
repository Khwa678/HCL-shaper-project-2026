import re

class EscalationEngine:
    """
    Multi-criteria decision logic for routing and human escalation.
    Evaluates confidence thresholds, intent severity, explicit agent requests,
    and multi-turn stalemate loops.
    """
    def __init__(self, confidence_threshold: float = 0.68, max_unresolved_turns: int = 5):
        self.confidence_threshold = confidence_threshold
        self.max_unresolved_turns = max_unresolved_turns
        
        # Explicit triggers for immediate human handoff
        self.explicit_triggers = [
            "human", "agent", "representative", "person", "operator",
            "manager", "supervisor", "lawyer", "sue", "unacceptable",
            "fraud", "scam", "grievance", "stolen"
        ]

    def evaluate(self, user_query: str, intent: str, confidence_score: float, 
                 current_turn_id: int, already_escalated: bool = False):
        """
        Determines whether the turn must escalate (1) or can be resolved by AI (0).
        Returns: (escalated_flag: int, reason: str)
        """
        # 1. Prior escalation inheritance
        if already_escalated:
            return 1, "Session previously transitioned to human agent care."

        query_lower = user_query.lower()

        # 2. Explicit human agent request
        for trigger in self.explicit_triggers:
            if re.search(r'\b' + re.escape(trigger) + r'\b', query_lower):
                return 1, f"Explicit escalation keyword detected: '{trigger}'."

        # 3. High severity intents with low-to-moderate confidence
        if intent in ["complaint", "cancellation"] and confidence_score < 0.80:
            return 1, f"High-risk intent '{intent}' requires senior agent intervention."

        # 4. Low NLU confidence threshold
        if confidence_score < self.confidence_threshold:
            return 1, f"NLU confidence ({confidence_score:.3f}) fell below reliability threshold ({self.confidence_threshold})."

        # 5. Conversation loop stalemate
        if current_turn_id >= self.max_unresolved_turns:
            return 1, f"Extended dialogue ({current_turn_id} turns) without quick resolution; escalating for customer convenience."

        # Normal automated resolution
        return 0, "Handled autonomously by AI Support Agent."
