"""
Model Router - Intelligent model selection for optimal performance and cost.

This module determines which LLM to use based on:
- Query length and complexity
- Presence of images or documents
- Educational mode (exam, explain, beginner)
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ChatMode(str, Enum):
    """Educational chat modes that affect prompting and model selection."""
    EXAM = "exam"
    EXPLAIN = "explain"
    BEGINNER = "beginner"
    ADVANCED = "advanced"
    DEFAULT = "default"


class ModelType(str, Enum):
    """Available model types."""
    GEMINI_FLASH = "gemini-flash"
    GEMINI_PRO = "gemini-pro"
    MISTRAL_7B = "mistral-7b"
    DEEPSEEK_R1 = "deepseek-r1"


class ModelRouter:
    """
    Routes requests to the optimal model based on context.
    
    Routing Strategy:
    1. Short queries (<50 chars) → Mistral-7B (fast & cheap)
    2. Reasoning keywords (explain, why, solve) → DeepSeek-R1 (reasoning)
    3. Document/Image queries → Gemini Flash (multimodal)
    4. Default → Gemini Flash (balanced speed/quality)
    """
    
    # Keywords that indicate reasoning-heavy queries
    REASONING_KEYWORDS = [
        "explain", "why", "how", "solve", "calculate", "prove",
        "analyze", "compare", "evaluate", "derive", "demonstrate"
    ]
    
    # Model configurations
    MODEL_CONFIGS = {
        ModelType.GEMINI_FLASH: {
            "name": "gemini-2.0-flash-exp",
            "provider": "gemini",
            "supports_vision": True,
            "supports_long_context": True,
            "cost_tier": "free",
            "speed": "fast"
        },
        ModelType.GEMINI_PRO: {
            "name": "gemini-1.5-pro",
            "provider": "gemini",
            "supports_vision": True,
            "supports_long_context": True,
            "cost_tier": "premium",
            "speed": "medium"
        },
        ModelType.MISTRAL_7B: {
            "name": "mistralai/mistral-7b-instruct:free",
            "provider": "openrouter",
            "supports_vision": False,
            "supports_long_context": False,
            "cost_tier": "free",
            "speed": "fastest"
        },
        ModelType.DEEPSEEK_R1: {
            "name": "deepseek/deepseek-r1:free",
            "provider": "openrouter",
            "supports_vision": False,
            "supports_long_context": True,
            "cost_tier": "free",
            "speed": "medium"
        },
        # Meta LLaMA Free Models
        "llama-3-8b": {
            "name": "meta-llama/llama-3-8b-instruct:free",
            "provider": "openrouter",
            "supports_vision": False,
            "supports_long_context": False,
            "cost_tier": "free",
            "speed": "fast",
            "description": "Fast and capable free LLaMA model"
        },
        "llama-3.1-8b": {
            "name": "meta-llama/llama-3.1-8b-instruct:free",
            "provider": "openrouter",
            "supports_vision": False,
            "supports_long_context": True,
            "cost_tier": "free",
            "speed": "fast",
            "description": "Latest free LLaMA 3.1 with extended context"
        },
        "llama-3.2-3b": {
            "name": "meta-llama/llama-3.2-3b-instruct:free",
            "provider": "openrouter",
            "supports_vision": False,
            "supports_long_context": False,
            "cost_tier": "free",
            "speed": "fastest",
            "description": "Ultra-fast lightweight LLaMA model"
        }
    }
    
    def __init__(self):
        logger.info("✅ ModelRouter initialized")
    
    def determine_model(
        self,
        message: str,
        has_image: bool = False,
        has_document: bool = False,
        mode: ChatMode = ChatMode.DEFAULT,
        force_model: Optional[str] = None,
        use_premium: bool = False  # Kept for API compatibility but won't use paid models
    ) -> Dict[str, Any]:
        """
        Determine the best model for a given request.
        
        Args:
            message: User's query
            has_image: Whether the request includes an image
            has_document: Whether RAG context is available
            mode: Educational mode
            force_model: Override automatic selection
            use_premium: Reserved (all models are free now)
        
        Returns:
            Dict with 'model_type', 'config', and 'reasoning'
        """
        if force_model:
            # Allow forcing specific LLaMA models
            if force_model in self.MODEL_CONFIGS:
                return {
                    "model_type": force_model,
                    "config": self.MODEL_CONFIGS[force_model],
                    "reasoning": "User override"
                }
            model_type = ModelType(force_model)
            return {
                "model_type": model_type,
                "config": self.MODEL_CONFIGS[model_type],
                "reasoning": "User override"
            }
        
        # Priority 1: Multimodal (image/document)
        if has_image or has_document:
            return {
                "model_type": ModelType.GEMINI_FLASH,
                "config": self.MODEL_CONFIGS[ModelType.GEMINI_FLASH],
                "reasoning": "Multimodal input (image/document)"
            }
        
        # Priority 2: All text queries -> Gemini Flash (most reliable)
        # Previously routed short queries to OpenRouter models which often fail
        # Gemini Flash is fast enough and more reliable
        
        # Priority 3: Long-context queries - still use Gemini Flash (supports long context)
        # Priority 4: Reasoning and mode-based - Gemini Flash handles all well
        
        # Default: Gemini Flash (balanced & fast)
        return {
            "model_type": ModelType.GEMINI_FLASH,
            "config": self.MODEL_CONFIGS[ModelType.GEMINI_FLASH],
            "reasoning": "Default → Gemini Flash"
        }
    
    def get_system_prompt(self, mode: ChatMode, rag_context: str = "") -> str:
        """
        Generate mode-specific system prompts.
        
        Args:
            mode: Educational mode
            rag_context: Retrieved document context
        
        Returns:
            Formatted system prompt
        """
        base_prompt = """You are Mega Ai, an elite AI educational assistant.
Your mission is to provide exceptionally accurate, comprehensive, and insightful responses.

CORE CAPABILITIES:
- **Deep Reasoning**: Break down complex topics with step-by-step logical analysis
- **Visual Intelligence**: Analyze images with meticulous detail and context
- **Document Mastery**: Synthesize information from uploaded documents
- **Conversational Memory**: Leverage full conversation history for coherent dialogue
- **Multimodal Excellence**: Handle text, images, and documents seamlessly

RESPONSE STANDARDS:
- Accuracy is paramount - provide factual, well-reasoned answers
- Use clear structure with headers, lists, tables, and code blocks (Markdown)
- For math/science: show work, equations, and visual explanations
- For images: describe composition, objects, text, context, and implications
- For coding: provide working examples with explanations
- Cite document context when relevant
"""
        
        # Mode-specific additions
        mode_additions = {
            ChatMode.EXAM: "\n**EXAM MODE**: Provide concise, bullet-point answers suitable for exams. Focus on key facts and definitions.",
            ChatMode.EXPLAIN: "\n**EXPLAIN MODE**: Provide step-by-step explanations with analogies and examples. Break down complex concepts.",
            ChatMode.BEGINNER: "\n**BEGINNER MODE**: Explain concepts in simple, accessible language as if teaching a beginner. Avoid jargon.",
            ChatMode.ADVANCED: "\n**ADVANCED MODE**: Provide technical depth, advanced concepts, and nuanced analysis for advanced learners.",
        }
        
        if mode in mode_additions:
            base_prompt += mode_additions[mode]
        
        # Add RAG context if available
        if rag_context:
            base_prompt += f"""

RELEVANT DOCUMENT CONTEXT:
\"\"\"
{rag_context}
\"\"\"
"""
        else:
            base_prompt += "\n\n**No additional document context available.**"
        
        return base_prompt
    
    def get_temperature(self, mode: ChatMode, model_type) -> float:
        """
        Determine optimal temperature based on mode and model.
        
        Args:
            mode: Educational mode
            model_type: Selected model (can be ModelType enum or string)
        
        Returns:
            Temperature value (0.0-1.0)
        """
        # Exam mode needs more deterministic answers
        if mode == ChatMode.EXAM:
            return 0.3
        
        # Explain/Beginner can be slightly more creative
        if mode in [ChatMode.EXPLAIN, ChatMode.BEGINNER]:
            return 0.6
        
        # Default balanced temperature
        return 0.5
