"""
Prompts for Resume Evaluation System

This module contains all the prompts used by the resume evaluation system.
Centralizing prompts here makes them easier to maintain and update.
"""

# Constants
DEFAULT_MODEL = "gemma3:4b"

# Model-specific parameters
MODEL_PARAMETERS = {
    "qwen3:1.7b": {
        "temperature": 0.0,
        "top_p": 0.9
    },
    "gemma3:1b": {
        "temperature": 0.0,
        "top_p": 0.9
    },    
    "qwen3:4b": {
        "temperature": 0.1,
        "top_p": 0.4
    },
    "gemma3:4b": {
        "temperature": 0.4,
        "top_p": 0.4
    },
    "mistral:7b": {
        "temperature": 0.0,
        "top_p": 0.9
    }
}





# JSON structure for evaluation response
EVALUATION_JSON_STRUCTURE = """Analyze the following resume and provide a JSON response with this EXACT structure (all fields are required):

{
    "candidate_name": "string (extract from resume)",
    "scores": {
        "open_source": {"score": 0, "max": 35, "evidence": "string"},
        "self_projects": {"score": 0, "max": 30, "evidence": "string"},
        "production": {"score": 0, "max": 25, "evidence": "string"},
        "technical_skills": {"score": 0, "max": 10, "evidence": "string"}
    },
    "bonus_points": {"total": 0, "breakdown": "string"},
    "deductions": {"total": 0, "reasons": "string"},
    "key_strengths": ["strength1", "strength2", "strength3"],
    "areas_for_improvement": ["improvement1", "improvement2"]
}

**CRITICAL REQUIREMENTS:**
1. You MUST respond with ONLY this JSON structure - no summary, no other fields
2. You MUST fill ALL FOUR score categories: open_source, self_projects, production, technical_skills
3. You MUST provide evidence for each score
4. You MUST NOT add any other fields like "summary", "skills", "experience", etc.
5. You MUST NOT change the field names or structure
6. The response must be valid JSON that matches this exact structure

**DO NOT RETURN A RESUME SUMMARY. RETURN ONLY THE SCORING EVALUATION IN THE SPECIFIED JSON FORMAT.**
"""
