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
        "temperature": 0.1,
        "top_p": 0.4
    },
    "mistral:7b": {
        "temperature": 0.0,
        "top_p": 0.9
    }
}

# JSON Resume extraction prompt
JSON_RESUME_EXTRACTION_PROMPT = """

--- The input markdown starts here ---

{text_content}

--- The input markdown ends here ---

**IMPORTANT**: Return ONLY valid JSON. Do not include any explanatory text, thinking process, or markdown formatting. The response must be a clean JSON object that can be parsed directly.
"""

JSON_RESUME_EXTRACTION_SYSTEM_MESSAGE = """You are an expert resume parser. Extract all information from resumes and format it according to the JSON Resume specification.

**CRITICAL: You must respond with ONLY valid JSON. Do not include any explanatory text, thinking process, markdown formatting, or <think> tags. Return ONLY the JSON object.**

Return a complete JSON Resume object with this structure:

{
  "basics": {
    "name": "Full name",
    "email": "Email address",
    "phone": "Phone number",
    "url": null,
    "summary": null,
    "location": {
      "city": "City",
      "countryCode": "Country code"
    },
    "profiles": [
      {
        "network": "Platform name (GitHub, LinkedIn, LeetCode, Stack Overflow, HackerRank)",
        "url": "Full URL",
        "username": "Username from URL"
      }
    ]
  },
  "education": [
    {
      "institution": "School/University name",
      "area": "Field of study",
      "studyType": "Degree type",
      "startDate": "Start date (YYYY-MM)",
      "endDate": "End date (YYYY-MM)",
      "score": "GPA/Percentage"
    }
  ],
  "work": [
    {
      "name": "Company name",
      "position": "Job title",
      "startDate": "Start date (YYYY-MM)",
      "endDate": "End date (YYYY-MM) or 'Present'",
      "summary": "Job description",
      "highlights": ["Achievement 1", "Achievement 2"]
    }
  ],
  "skills": [
    {
      "name": "Skill category",
      "level": null,
      "keywords": ["Skill 1", "Skill 2"]
    }
  ],
  "projects": [
    {
      "name": "Project name",
      "description": "Project description",
      "url": "Project URL",
      "technologies": ["Tech 1", "Tech 2"]
    }
  ],
  "awards": [
    {
      "title": "Award name",
      "date": "Award date (YYYY-MM)",
      "awarder": "Awarding organization"
    }
  ]
}

**EXTRACTION RULES:**
1. **PROFILES**: Extract ALL profiles from "LINKS" section - look for URLs like github.com, linkedin.com, leetcode.com, stackoverflow.com, hackerrank.com
2. **WORK**: Extract from "EXPERIENCE" section
3. **SKILLS**: Extract from "SKILLS" section
4. **PROJECTS**: Extract from "PROJECTS" section
5. **AWARDS**: Extract from "HONORS & AWARDS" section
6. **EDUCATION**: Extract from "EDUCATION" section
7. Convert dates to YYYY-MM format
8. Use "Present" for ongoing positions

**CRITICAL**: You MUST extract profiles from the "LINKS" section. Do not skip this section.

**IMPORTANT EXTRACTION DETAILS:**
1. Extract email from mailto: links (e.g., [email](mailto:email@example.com))
2. Extract phone number separately from contact information
3. Extract usernames from profile URLs (e.g., from https://github.com/username, extract "username")
4. Parse education dates properly (e.g., "June 2018 - June 2022" should be startDate: "2018-06", endDate: "2022-06")
5. Extract work experience from "EXPERIENCE" section with proper company names and positions
6. Extract awards from "HONORS & AWARDS" section
"""