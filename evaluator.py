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
                    'temperature': self.model_params['temperature'],
                    'top_p': self.model_params['top_p']
                },
                format= EvaluationData.model_json_schema()
            )

            response_text = response['message']['content']

            logger.info(f"🔤 Prompt response: {response_text}")

            return (response_text)

            evaluation_data = EvaluationData(**evaluation_dict)

            return evaluation_data

        except Exception as e:
            print(f"Error evaluating resume: {str(e)}")
            raise

    def _convert_json_resume_to_text(self, resume_data: JSONResume) -> str:
        text_parts = []

        if resume_data.basics:
            basics = resume_data.basics
            text_parts.append("=== BASIC INFORMATION ===")
            text_parts.append(f"Name: {basics.name or 'Not provided'}")
            text_parts.append(f"Email: {basics.email or 'Not provided'}")
            text_parts.append(f"Phone: {basics.phone or 'Not provided'}")
            text_parts.append(f"Website: {basics.url or 'Not provided'}")

            if basics.summary:
                text_parts.append(f"Summary: {basics.summary}")

            if basics.location:
                loc = basics.location
                location_parts = []
                if loc.address:
                    location_parts.append(loc.address)
                if loc.city:
                    location_parts.append(loc.city)
                if loc.region:
                    location_parts.append(loc.region)
                if loc.postalCode:
                    location_parts.append(loc.postalCode)
                if loc.countryCode:
                    location_parts.append(loc.countryCode)

                if location_parts:
                    text_parts.append(f"Location: {', '.join(location_parts)}")

            if basics.profiles:
                text_parts.append("Profiles:")
                for profile in basics.profiles:
                    text_parts.append(f"  - {profile.network}: {profile.username} ({profile.url})")

        if resume_data.work:
            text_parts.append("\n=== WORK EXPERIENCE ===")
            for i, work in enumerate(resume_data.work, 1):
                text_parts.append(f"{i}. {work.position} at {work.name}")
                text_parts.append(f"   Period: {work.startDate} - {work.endDate}")
                if work.url:
                    text_parts.append(f"   Website: {work.url}")
                if work.summary:
                    text_parts.append(f"   Description: {work.summary}")
                if work.highlights:
                    text_parts.append("   Key Achievements:")
                    for highlight in work.highlights:
                        text_parts.append(f"     • {highlight}")

        if resume_data.education:
            text_parts.append("\n=== EDUCATION ===")
            for i, edu in enumerate(resume_data.education, 1):
                text_parts.append(f"{i}. {edu.studyType} in {edu.area}")
                text_parts.append(f"   Institution: {edu.institution}")
                text_parts.append(f"   Period: {edu.startDate} - {edu.endDate}")
                if edu.score:
                    text_parts.append(f"   Score: {edu.score}")
                if edu.url:
                    text_parts.append(f"   Website: {edu.url}")
                if edu.courses:
                    text_parts.append(f"   Courses: {', '.join(edu.courses)}")

        if resume_data.skills:
            text_parts.append("\n=== SKILLS ===")
            for skill in resume_data.skills:
                text_parts.append(f"• {skill.name}")
                if skill.level:
                    text_parts.append(f"  Level: {skill.level}")
                if skill.keywords:
                    text_parts.append(f"  Keywords: {', '.join(skill.keywords)}")

        if resume_data.projects:
            text_parts.append("\n=== PROJECTS ===")
            for i, project in enumerate(resume_data.projects, 1):
                text_parts.append(f"{i}. {project.name}")
                if project.startDate and project.endDate:
                    text_parts.append(f"   Period: {project.startDate} - {project.endDate}")
                if project.description:
                    text_parts.append(f"   Description: {project.description}")
                if project.url:
                    text_parts.append(f"   URL: {project.url}")
                if project.highlights:
                    text_parts.append("   Highlights:")
                    for highlight in project.highlights:
                        text_parts.append(f"     • {highlight}")

        if resume_data.awards:
            text_parts.append("\n=== AWARDS ===")
            for award in resume_data.awards:
                text_parts.append(f"• {award.title} - {award.awarder} ({award.date})")
                if award.summary:
                    text_parts.append(f"  {award.summary}")

        if resume_data.certificates:
            text_parts.append("\n=== CERTIFICATES ===")
            for cert in resume_data.certificates:
                text_parts.append(f"• {cert.name} - {cert.issuer} ({cert.date})")
                if cert.url:
                    text_parts.append(f"  URL: {cert.url}")

        if resume_data.publications:
            text_parts.append("\n=== PUBLICATIONS ===")
            for pub in resume_data.publications:
                text_parts.append(f"• {pub.name} - {pub.publisher} ({pub.releaseDate})")
                if pub.url:
                    text_parts.append(f"  URL: {pub.url}")
                if pub.summary:
                    text_parts.append(f"  {pub.summary}")

        if resume_data.languages:
            text_parts.append("\n=== LANGUAGES ===")
            for lang in resume_data.languages:
                text_parts.append(f"• {lang.language} - {lang.fluency}")

        if resume_data.interests:
            text_parts.append("\n=== INTERESTS ===")
            for interest in resume_data.interests:
                text_parts.append(f"• {interest.name}")
                if interest.keywords:
                    text_parts.append(f"  Keywords: {', '.join(interest.keywords)}")

        if resume_data.references:
            text_parts.append("\n=== REFERENCES ===")
            for ref in resume_data.references:
                text_parts.append(f"• {ref.name}")
                if ref.reference:
                    text_parts.append(f"  {ref.reference}")

        if resume_data.volunteer:
            text_parts.append("\n=== VOLUNTEER EXPERIENCE ===")
            for volunteer in resume_data.volunteer:
                text_parts.append(f"• {volunteer.position} at {volunteer.organization}")
                text_parts.append(f"  Period: {volunteer.startDate} - {volunteer.endDate}")
                if volunteer.url:
                    text_parts.append(f"  Website: {volunteer.url}")
                if volunteer.summary:
                    text_parts.append(f"  Description: {volunteer.summary}")
                if volunteer.highlights:
                    text_parts.append("  Highlights:")
                    for highlight in volunteer.highlights:
                        text_parts.append(f"    • {highlight}")

        return "\n".join(text_parts)

    def _convert_github_data_to_text(self, github_data: dict) -> str:
        github_text = "\n\n=== GITHUB DATA ===\n"

        if 'profile' in github_data:
            profile = github_data['profile']
            github_text += f"GitHub Profile:\n"
            github_text += f"- Username: {profile.get('username', 'N/A')}\n"
            github_text += f"- Name: {profile.get('name', 'N/A')}\n"
            github_text += f"- Bio: {profile.get('bio', 'N/A')}\n"
            github_text += f"- Public Repositories: {profile.get('public_repos', 'N/A')}\n"
            github_text += f"- Followers: {profile.get('followers', 'N/A')}\n"
            github_text += f"- Following: {profile.get('following', 'N/A')}\n"
            github_text += f"- Account Created: {profile.get('created_at', 'N/A')}\n"
            github_text += f"- Last Updated: {profile.get('updated_at', 'N/A')}\n"

        if 'projects' in github_data:
            projects = github_data['projects']
            github_text += f"\nGitHub Projects ({len(projects)} total):\n"
            for i, project in enumerate(projects[:10], 1):
                github_text += f"{i}. {project.get('name', 'N/A')}\n"
                github_text += f"   Description: {project.get('description', 'N/A')}\n"
                github_text += f"   URL: {project.get('github_url', 'N/A')}\n"
                if 'github_details' in project:
                    details = project['github_details']
                    github_text += f"   Stars: {details.get('stars', 'N/A')}\n"
                    github_text += f"   Forks: {details.get('forks', 'N/A')}\n"
                    github_text += f"   Language: {details.get('language', 'N/A')}\n"
                github_text += "\n"

        return github_text

    def _convert_blog_data_to_text(self, blog_data: dict) -> str:
        blog_text = "\n\n=== BLOG DATA ===\n"
        blog_text += f"Total Blogs Found: {blog_data.get('total_blogs', 'N/A')}\n"
        blog_text += f"Blog Score: {blog_data.get('blog_score', 'N/A')}/10.0\n"
        blog_text += f"Blog Details: {blog_data.get('blog_details', 'N/A')}\n"

        if 'blogs' in blog_data:
            blog_text += "\nBlog URLs Found:\n"
            for i, blog in enumerate(blog_data['blogs'][:5], 1):
                blog_text += f"{i}. {blog.get('url', 'N/A')}\n"
                blog_text += f"   Score: {blog.get('score', 'N/A')}/10.0\n"
                blog_text += f"   Details: {blog.get('details', 'N/A')}\n"
                blog_text += "\n"

        return blog_text
