from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator
from models import JSONResume, EvaluationData
import logging
import ollama
import json
import re

MAX_BONUS_POINTS = 20
MIN_FINAL_SCORE = -20
MAX_FINAL_SCORE = 120

from prompt import DEFAULT_MODEL
from prompts.template_manager import TemplateManager

logger = logging.getLogger(__name__)


class ResumeEvaluator:
    def __init__(self, model_name: str = DEFAULT_MODEL, model_params: dict = None):
        if not model_name:
            raise ValueError("Model name cannot be empty")

        self.model_name = model_name
        self.model_params = model_params
        self.template_manager = TemplateManager()

    def _load_evaluation_prompt(self, resume_text: str) -> str:
        criteria_template = self.template_manager.render_resume_evaluation_criteria_template(resume_text)
        if criteria_template is None:
            raise ValueError("Failed to load resume evaluation criteria template")
        return criteria_template

    def evaluate_resume(self, resume_text: str) -> EvaluationData:
        self._last_resume_text = resume_text

        full_prompt = self._load_evaluation_prompt(resume_text)

        # logger.info(f"🔤 Evaluation prompt being sent: {full_prompt}")

        try:
            system_message = self.template_manager.render_resume_evaluation_system_message_template()
            if system_message is None:
                raise ValueError("Failed to load resume evaluation system message template")
                
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': system_message
                    },
                    {
                        'role': 'user',
                        'content': full_prompt
                    }
                ],
                options={
                    'temperature': 0.5
                },
                format= EvaluationData.model_json_schema()
            )

            response_text = response['message']['content']

            logger.debug(f"🔤 Prompt response: {response_text}")

            evaluation_dict = json.loads(response_text)
            evaluation_data = EvaluationData(**evaluation_dict)

            return evaluation_data

        except Exception as e:
            print(f"Error evaluating resume: {str(e)}")
            raise



    
