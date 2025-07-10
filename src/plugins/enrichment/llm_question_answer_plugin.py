"""
LLM Question Answer Plugin
HTTP-only plugin that calls LLM service for question answering.
"""

import logging
from typing import Dict, Any, List

from src.plugins.http_base import HTTPBasePlugin
from src.plugins.base import PluginMetadata, PluginResourceRequirements, PluginType, ExecutionPriority

logger = logging.getLogger(__name__)


class LLMQuestionAnswerPlugin(HTTPBasePlugin):
    """
    HTTP-only plugin that performs question answering using ONLY LLM.
    
    Features:
    - Calls LLM service endpoint for Q&A operations
    - Context-aware question answering
    - Multiple question formats supported
    - No provider management - just HTTP calls
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="LLMQuestionAnswerPlugin",
            version="1.0.0",
            description="Performs question answering using LLM",
            author="System",
            plugin_type=PluginType.EMBED_DATA_EMBELLISHER,
            tags=["qa", "question", "answer", "llm", "ai", "reasoning"],
            execution_priority=ExecutionPriority.HIGH
        )
    
    @property
    def resource_requirements(self) -> PluginResourceRequirements:
        """LLM Q&A plugins have higher resource requirements due to model inference."""
        return PluginResourceRequirements(
            min_cpu_cores=1.0,
            preferred_cpu_cores=2.0,
            min_memory_mb=512.0,
            preferred_memory_mb=2048.0,
            requires_gpu=True,
            min_gpu_memory_mb=2048.0,
            preferred_gpu_memory_mb=8192.0,
            max_execution_time_seconds=90.0
        )
    
    async def enrich_field(
        self, 
        field_name: str, 
        field_value: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enrich a field with LLM question answering.
        
        Args:
            field_name: Name of the field being enriched
            field_value: Value of the field (used as context)
            config: Plugin configuration including questions
            
        Returns:
            Dict containing LLM Q&A results
        """
        try:

            if isinstance(field_value, str):
                context = field_value
            elif isinstance(field_value, list):
                context = " ".join(str(item) for item in field_value)
            else:
                context = str(field_value)
            
            
            questions = config.get("questions", [])
            if not questions:

                questions = self._get_default_questions(field_name)
            
            if not questions:
                self._logger.debug(f"No questions provided for field {field_name}")
                result = {"llm_qa": []}
                
                return self.normalize_text(result)
            
            self._logger.debug(f"Answering {len(questions)} questions for field {field_name}")
            
            

            service_url = await self.get_plugin_service_url()
            
            
            concept_text = f"Context: {context}\n\nQuestions: {', '.join(questions)}"
            
            request_data = {
                "concept": concept_text,
                "media_context": "movie",
                "max_concepts": config.get("max_concepts", 10),
                "field_name": field_name,
                "options": {
                    "task_type": "question_answering",
                    "questions": questions,
                    "answer_format": config.get("answer_format", "concise"),
                    "max_answer_length": config.get("max_answer_length", 200),
                    "confidence_threshold": config.get("confidence_threshold", 0.5),
                    "temperature": config.get("temperature", 0.1),
                    "include_reasoning": config.get("include_reasoning", False)
                }
            }
            
            response = await self.http_post(service_url, request_data)
            
            
            if response.get("success", False):
                result_data = response.get("result", {})

                qa_results = result_data.get("expanded_concepts", 
                           result_data.get("answers", 
                           result_data.get("concepts", [])))
                metadata = response.get("metadata", {})
            else:
                qa_results = []
                metadata = {"error": response.get("error_message", "Unknown error")}
            
            self._logger.info(
                f"LLM answered {len(qa_results)} questions for field {field_name}"
            )
            
            result = {
                "llm_qa": qa_results,
                "context": context[:300] + "..." if len(context) > 300 else context,
                "questions": questions,
                "field_name": field_name,
                "metadata": {
                    "provider": "llm_qa",
                    "question_count": len(questions),
                    "answer_count": len(qa_results),
                    "context_length": len(context),
                    "service_metadata": metadata,
                    "answer_format": config.get("answer_format", "concise")
                }
            }
            
            
            return self.normalize_text(result)
            
        except Exception as e:
            self._logger.error(f"LLM Q&A failed for field {field_name}: {e}")
            
            result = {
                "llm_qa": [],
                "context": "",
                "questions": [],
                "field_name": field_name,
                "error": str(e),
                "metadata": {
                    "provider": "llm_qa",
                    "success": False
                }
            }
            
            
            return self.normalize_text(result)
    
    def _get_default_questions(self, field_name: str) -> List[str]:
        """Get default questions based on field name."""
        field_questions = {
            "overview": [
                "What is the main theme or topic?",
                "What genre does this belong to?",
                "What are the key elements or features?"
            ],
            "description": [
                "What is this about?",
                "What are the main characteristics?",
                "What makes this unique or notable?"
            ],
            "plot": [
                "What is the main storyline?",
                "Who are the main characters?",
                "What is the central conflict or theme?"
            ],
            "summary": [
                "What are the key points?",
                "What is the main message?",
                "What should someone know about this?"
            ],
            "content": [
                "What type of content is this?",
                "What is the main subject matter?",
                "What audience is this for?"
            ]
        }
        

        field_lower = field_name.lower()
        for key, questions in field_questions.items():
            if key in field_lower:
                return questions
        

        return [
            "What is this about?",
            "What are the main themes or topics?",
            "What category does this belong to?"
        ]
    
    async def answer_questions(
        self, 
        context: str,
        questions: List[str], 
        config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Direct Q&A method for backwards compatibility.
        
        Args:
            context: Context text for answering questions
            questions: List of questions to answer
            config: Q&A configuration
            
        Returns:
            Q&A results
        """
        if config is None:
            config = {}
            
        config["questions"] = questions
        
        
        result = await self.enrich_field("context", context, config)
        response = {
            "answers": result.get("llm_qa", []),
            "metadata": result.get("metadata", {})
        }
        
        
        return self.normalize_text(response)
    
    async def answer_single_question(
        self,
        context: str,
        question: str,
        answer_format: str = "concise"
    ) -> Dict[str, Any]:
        """
        Answer a single question with context.
        
        Args:
            context: Context text for answering
            question: Single question to answer
            answer_format: Format of answer (concise, detailed, bullet_points)
            
        Returns:
            Single answer result
        """
        config = {
            "questions": [question],
            "answer_format": answer_format,
            "include_reasoning": True
        }
        
        result = await self.enrich_field("single_qa", context, config)
        qa_results = result.get("llm_qa", [])
        
        response = {
            "question": question,
            "answer": qa_results[0] if qa_results else None,
            "context": context[:300] + "..." if len(context) > 300 else context,
            "metadata": result.get("metadata", {})
        }
        
        
        return self.normalize_text(response)
    
    async def extract_key_information(
        self,
        text: str,
        information_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Extract key information using targeted questions.
        
        Args:
            text: Text to extract information from
            information_type: Type of information (general, narrative, technical, biographical)
            
        Returns:
            Extracted key information
        """

        question_sets = {
            "general": [
                "What is the main topic or subject?",
                "What are the key facts or details?",
                "What is the significance or importance?"
            ],
            "narrative": [
                "What is the main storyline or plot?",
                "Who are the main characters or people involved?",
                "When and where does this take place?",
                "What is the outcome or resolution?"
            ],
            "technical": [
                "What is the main technology or method described?",
                "What are the key specifications or requirements?",
                "What are the benefits or applications?"
            ],
            "biographical": [
                "Who is the main person?",
                "What are their key achievements or contributions?",
                "When and where did key events occur?"
            ]
        }
        
        questions = question_sets.get(information_type, question_sets["general"])
        
        config = {
            "questions": questions,
            "answer_format": "detailed",
            "include_reasoning": False
        }
        
        result = await self.enrich_field("extraction", text, config)
        
        response = {
            "information_type": information_type,
            "key_information": result.get("llm_qa", []),
            "metadata": result.get("metadata", {})
        }
        
        
        return self.normalize_text(response)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check plugin and service health."""
        base_health = await super().health_check()
        
        try:
            
            service_url = self.get_service_url("llm", "health")
            health_response = await self.http_get(service_url)
            
            base_health["service_health"] = {
                "llm_qa_service": "healthy",
                "service_response": health_response
            }
            
        except Exception as e:
            base_health["service_health"] = {
                "llm_qa_service": "unhealthy",
                "error": str(e)
            }
        
        return base_health