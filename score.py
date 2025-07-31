import os
import sys
import json
import logging
from pdf import PDFHandler
from github import fetch_and_display_github_info
from models import JSONResume, EvaluationData
from typing import List, Optional, Dict
from evaluator import ResumeEvaluator
from pathlib import Path
from prompt import DEFAULT_MODEL, MODEL_PARAMETERS

# Development mode flag - set to False for production
DEVELOPMENT_MODE = True

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)5s - %(lineno)5d - %(funcName)33s - %(levelname)5s - %(message)s'
)

def _evaluate_resume(resume_data: JSONResume, github_data: dict = None, blog_data: dict = None) -> Optional[EvaluationData]:
    """Evaluate the resume using AI and display results."""
    
    model_params = MODEL_PARAMETERS.get(DEFAULT_MODEL)
    evaluator = ResumeEvaluator(model_name=DEFAULT_MODEL, model_params=model_params)

    # Convert JSON resume data to text
    resume_text = evaluator._convert_json_resume_to_text(resume_data)

    # Add GitHub data if available
    if github_data:
        github_text = evaluator._convert_github_data_to_text(github_data)
        resume_text += github_text

    # Add blog data if available
    if blog_data:
        blog_text = evaluator._convert_blog_data_to_text(blog_data)
        resume_text += blog_text

    # Evaluate the enhanced resume
    evaluation_result = evaluator.evaluate_resume(resume_text)

    # print(evaluation_result)

    return evaluation_result

def find_profile(profiles, network):
        return next((p for p in profiles if p.network and p.network.lower() == network.lower()), None)

def main(pdf_path):
    # Create cache filename based on PDF path
    cache_filename = f"cache/resumecache_{os.path.basename(pdf_path).replace('.pdf', '')}.json"
    github_cache_filename = f"cache/githubcache_{os.path.basename(pdf_path).replace('.pdf', '')}.json"

    # Check if cache exists and we're in development mode
    if DEVELOPMENT_MODE and os.path.exists(cache_filename):
        print(f"Loading cached data from {cache_filename}")
        cached_data = json.loads(Path(cache_filename).read_text())
        resume_data = JSONResume(**cached_data)
    else:
        print(f"Extracting data from PDF" + (" and caching to " + cache_filename if DEVELOPMENT_MODE else ""))
        pdf_handler = PDFHandler()
        resume_data = pdf_handler.extract_json_from_pdf(pdf_path)
        if DEVELOPMENT_MODE:
            Path(cache_filename).write_text(json.dumps(resume_data.model_dump(), indent=2, ensure_ascii=False))
    
    # Check if cache exists and we're in development mode
    github_data = {}
    if DEVELOPMENT_MODE and os.path.exists(github_cache_filename):
        print(f"Loading cached data from {github_cache_filename}")
        github_data = json.loads(Path(github_cache_filename).read_text())
    else:
        print(f"Fetching GitHub data" + (" and caching to " + github_cache_filename if DEVELOPMENT_MODE else ""))
        profiles = resume_data.basics.profiles
        github_profile = find_profile(profiles, "Github")

        if github_profile:
            github_data = fetch_and_display_github_info(github_profile.url)
        if DEVELOPMENT_MODE:
            Path(github_cache_filename).write_text(json.dumps(github_data, indent=2, ensure_ascii=False))
        
    score = _evaluate_resume(resume_data, github_data)
    return score

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python score.py <pdf_path>")
        exit(1)
    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"Error: File '{pdf_path}' does not exist.")
        exit(1)

    main(pdf_path)
