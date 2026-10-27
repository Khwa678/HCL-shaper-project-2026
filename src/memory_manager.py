from datetime import datetime
from typing import List, Dict, Any

class ConversationMemoryManager:
    """
    Stateful conversational memory manager keyed by session_id.
    Maintains dialogue history, turn sequences, and conversation buffer
    with a sliding context window to prevent LLM token budget overflow.
    """
    def __init__(self, max_history_turns: int = 6):
        self.max_history_turns = max_history_turns
        self.sessions: Dict[str, List[Dict[str, Any]]] = {}

    def get_or_create_session(self, session_id: str) -> List[Dict[str, Any]]:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]

    def add_turn(self, session_id: str, user_query: str, bot_response: str, 
                 intent: str, confidence_score: float, escalated: int, 
                 response_time_ms: int, satisfaction_score: int = None) -> int:
        """
        Appends a conversational turn and returns the sequential turn_id.
        """
        history = self.get_or_create_session(session_id)
        turn_id = len(history) + 1
        
        turn_data = {
            'turn_id': turn_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user_query': user_query,
            'bot_response': bot_response,
            'intent': intent,
            'confidence_score': confidence_score,
            'escalated': escalated,
            'response_time_ms': response_time_ms,
            'satisfaction_score': satisfaction_score
        }
        history.append(turn_data)
        return turn_id

    def get_turn_count(self, session_id: str) -> int:
        return len(self.sessions.get(session_id, []))

    def get_windowed_dialogue(self, session_id: str) -> str:
        """
        Returns a formatted string of the last N turns for LLM prompt context injection.
        """
        history = self.sessions.get(session_id, [])
        if not history:
            return "No previous conversation history."

        # Take last N turns
        recent_turns = history[-self.max_history_turns:]
        formatted = []
        for turn in recent_turns:
            formatted.append(f"Customer (Turn {turn['turn_id']}): {turn['user_query']}")
            formatted.append(f"Assistant: {turn['bot_response']}")

        return "\n".join(formatted)

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        return self.sessions.get(session_id, [])

    def is_session_escalated(self, session_id: str) -> bool:
        history = self.sessions.get(session_id, [])
        return any(turn.get('escalated') == 1 for turn in history)

    def set_satisfaction_score(self, session_id: str, score: int):
        history = self.sessions.get(session_id, [])
        if history:
            # Update the latest turn
            history[-1]['satisfaction_score'] = score
