import os
import time
from typing import Dict, Any, Optional

from src.intent_router import IntentRouter
from src.rag_engine import RAGEngine
from src.memory_manager import ConversationMemoryManager
from src.escalation_engine import EscalationEngine

class SupportChatbotOrchestrator:
    """
    Main Orchestration Engine combining:
    1. LangChain Conversation Memory
    2. NLU Intent Classification across 10 categories
    3. RAG Knowledge Base Vector Retrieval
    4. Guardrail & Multi-factor Escalation Logic
    5. OpenAI / Local Contextual LLM Response Generation
    6. Ethical AI Identity Disclosure
    """
    def __init__(self, openai_api_key: Optional[str] = None):
        self.intent_router = IntentRouter()
        self.rag_engine = RAGEngine()
        self.memory = ConversationMemoryManager(max_history_turns=6)
        self.escalation_engine = EscalationEngine(confidence_threshold=0.68, max_unresolved_turns=5)
        
        # Check for OpenAI Key
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm_available = False
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                self.llm_available = True
            except Exception as e:
                print(f"OpenAI initialization skipped: {e}")

    def process_message(self, session_id: str, user_query: str) -> Dict[str, Any]:
        """
        End-to-end processing pipeline for a customer query turn.
        """
        start_time = time.time()
        
        # 1. Inspect session history & turn sequence
        history = self.memory.get_session_history(session_id)
        current_turn_id = len(history) + 1
        already_escalated = self.memory.is_session_escalated(session_id)

        # 2. Intent Routing & Confidence Scoring
        intent, confidence_score = self.intent_router.classify(user_query)

        # 3. RAG Context Retrieval
        retrieved_docs = self.rag_engine.retrieve(user_query, category_filter=intent, top_k=2)
        formatted_context = self.rag_engine.format_context(retrieved_docs)

        # 4. Escalation Evaluation
        escalated, escalation_reason = self.escalation_engine.evaluate(
            user_query=user_query,
            intent=intent,
            confidence_score=confidence_score,
            current_turn_id=current_turn_id,
            already_escalated=already_escalated
        )

        # 5. Response Synthesis
        if escalated:
            bot_response = (
                f"I am routing your request to a Senior Human Support Specialist ({escalation_reason}). "
                "An agent has been notified and will review your conversation history. "
                "In the meantime, our knowledge base notes: " + (retrieved_docs[0]['answer'] if retrieved_docs else "Please hold on.")
            )
        elif self.llm_available and self.client:
            bot_response = self._generate_openai_response(session_id, user_query, formatted_context, intent)
        else:
            bot_response = self._generate_local_contextual_response(user_query, retrieved_docs, intent)

        # Calculate latency in ms
        elapsed_ms = int((time.time() - start_time) * 1000)
        # Ensure simulated natural typing latency of at least 250ms
        response_time_ms = max(250, elapsed_ms)

        # 6. Record turn into Conversation Memory
        self.memory.add_turn(
            session_id=session_id,
            user_query=user_query,
            bot_response=bot_response,
            intent=intent,
            confidence_score=confidence_score,
            escalated=escalated,
            response_time_ms=response_time_ms
        )

        return {
            'session_id': session_id,
            'turn_id': current_turn_id,
            'user_query': user_query,
            'bot_response': bot_response,
            'intent': intent,
            'confidence_score': confidence_score,
            'escalated': escalated,
            'escalation_reason': escalation_reason,
            'retrieved_context': retrieved_docs,
            'response_time_ms': response_time_ms
        }

    def _generate_openai_response(self, session_id: str, query: str, context: str, intent: str) -> str:
        """
        Uses OpenAI Chat Completion with conversation context and RAG grounding.
        """
        dialogue_history = self.memory.get_windowed_dialogue(session_id)
        system_prompt = (
            "You are Nova, an AI-powered Enterprise Customer Support Assistant. "
            "Always disclose your AI identity when greeted. "
            "Be empathetic, direct, concise, and helpful. "
            "Use the provided Knowledge Base Context to answer the user question accurately. "
            "If the context does not contain the answer, politely offer to escalate to human support.\n\n"
            f"KNOWLEDGE BASE CONTEXT:\n{context}\n\n"
            f"CONVERSATION HISTORY:\n{dialogue_history}"
        )
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.2,
                max_tokens=250
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to local synthesis if API call fails
            return self._generate_local_contextual_response(query, [], intent)

    def _generate_local_contextual_response(self, query: str, retrieved_docs: list, intent: str) -> str:
        """
        Deterministic, domain-accurate local RAG response generator.
        Ensures 100% offline functionality without API keys or token cost.
        """
        # Ethical identity disclosure if greeting or introductory query
        if any(w in query.lower() for w in ["hello", "hi", "hey", "who are you", "help"]):
            intro = "Hello! I am Nova, an AI Customer Support Assistant. "
        else:
            intro = ""

        if retrieved_docs and retrieved_docs[0]['similarity_score'] > 0.08:
            top_match = retrieved_docs[0]
            answer = (
                f"{intro}Based on our knowledge base for **{top_match['category'].replace('_', ' ').title()}**:\n\n"
                f"{top_match['answer']}\n\n"
                f"*(Reference: FAQ {top_match['id']} — Match score: {top_match['similarity_score']:.2f})*"
            )
            return answer
        else:
            return (
                f"{intro}I understand you have an inquiry regarding **{intent.replace('_', ' ').title()}**. "
                "I am scanning our records to provide the most accurate assistance. "
                "Could you provide a few additional details (such as an order ID or account email) so I can resolve this, "
                "or type 'agent' if you would prefer to speak directly with our team?"
            )
