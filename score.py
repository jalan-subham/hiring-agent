import os
import json
from pdf import PDFHandler
from github import fetch_and_display_github_info
from models import JSONResume, EvaluationResult
from typing import List, Optional, Dict
from evaluator import ResumeEvaluator
from pathlib import Path
from prompt import DEFAULT_MODEL

def _evaluate_resume(resume_data: JSONResume, github_data: dict = None, blog_data: dict = None) -> Optional[EvaluationResult]:
    """Evaluate the resume using AI and display results."""
    evaluator = ResumeEvaluator(model_name=DEFAULT_MODEL, temperature=0.2)

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

    report = evaluator.format_evaluation_report(evaluation_result)
    print(report)

    return evaluation_result

def find_profile(profiles, network):
        return next((p for p in profiles if p.network and p.network.lower() == network.lower()), None)

if __name__ == "__main__":
    pdf_path = "resume/Ajitesh_Panda.pdf"
    
    # Create cache filename based on PDF path
    cache_filename = f"cache/resumecache_{os.path.basename(pdf_path).replace('.pdf', '')}.json"
    github_cache_filename = f"cache/githubcache_{os.path.basename(pdf_path).replace('.pdf', '')}.json"

    # Check if cache exists
    if os.path.exists(cache_filename):
        print(f"Loading cached data from {cache_filename}")
        cached_data = json.loads(Path(cache_filename).read_text())
        resume_data = JSONResume(**cached_data)
    else:
        print(f"Extracting data from PDF and caching to {cache_filename}")
        pdf_handler = PDFHandler()
        resume_data = pdf_handler.extract_json_from_pdf(pdf_path)
        Path(cache_filename).write_text(json.dumps(resume_data.model_dump(), indent=2, ensure_ascii=False))
    
    # Check if cache exists
    github_data = {}
    if os.path.exists(github_cache_filename):
        print(f"Loading cached data from {github_cache_filename}")
        github_data = json.loads(Path(github_cache_filename).read_text())
    else:
        print(f"Extracting data from PDF and caching to {github_cache_filename}")
        profiles = resume_data.basics.profiles
        github_profile = find_profile(profiles, "Github")

        if github_profile:
            github_data = fetch_and_display_github_info(github_profile.url)
        Path(github_cache_filename).write_text(json.dumps(github_data, indent=2, ensure_ascii=False))
        
    score = _evaluate_resume(resume_data, github_data)