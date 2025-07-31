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

from prompt import (
    DEFAULT_MODEL,
    EVALUATION_JSON_STRUCTURE,
    RESUME_EVALUATION_SYSTEM_MESSAGE,
)
from prompts.template_manager import TemplateManager

logger = logging.getLogger(__name__)

class EvaluationResult(BaseModel):
    candidate_name: str
    total_score: float = Field(ge=0, le=100)
    final_score: float
    category_breakdown: Dict[str, Tuple[float, str]]
    scoring_explanation: str
    key_strengths: List[str]
    areas_for_improvement: List[str]

    class Config:
        use_enum_values = True

    @field_validator('final_score')
    @classmethod
    def validate_final_score(cls, v, info):
        if info.data and 'total_score' in info.data:
            if v < MIN_FINAL_SCORE or v > MAX_FINAL_SCORE:
                raise ValueError(f'Final score must be between {MIN_FINAL_SCORE} and {MAX_FINAL_SCORE}')
        return v

    def to_json(self) -> str:
        return self.json(indent=2)

    def to_dict(self) -> dict:
        return self.dict()


class ResumeEvaluator:
    def __init__(self, model_name: str = DEFAULT_MODEL, temperature: float = 0.2):
        if not model_name:
            raise ValueError("Model name cannot be empty")
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")

        self.model_name = model_name
        self.temperature = temperature
        self.template_manager = TemplateManager()
        self.evaluation_prompt = self._load_evaluation_prompt()

    def _load_evaluation_prompt(self) -> str:
        criteria_template = self.template_manager.render_resume_evaluation_criteria_template()
        if criteria_template is None:
            raise ValueError("Failed to load resume evaluation criteria template")
        return criteria_template + EVALUATION_JSON_STRUCTURE + "\nResume to evaluate:\n"

    def evaluate_resume(self, resume_text: str) -> EvaluationResult:
        self._last_resume_text = resume_text

        full_prompt = self.evaluation_prompt + resume_text

        # logger.info(f"🔤 Evaluation prompt being sent: {full_prompt}")

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': RESUME_EVALUATION_SYSTEM_MESSAGE
                    },
                    {
                        'role': 'user',
                        'content': full_prompt
                    }
                ],
                options={
                    'temperature': self.temperature,
                    'top_p': 0.9
                },
                format= EvaluationData.model_json_schema()
            )

            response_text = response['message']['content']

            logger.info(f"🔤 Prompt response: {response_text}")

            return (response_text)

            evaluation_data = EvaluationData(**evaluation_dict)

            return self._create_evaluation_result(evaluation_data)

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

    def _generate_strengths_from_scores(self, scores: dict) -> List[str]:
        strengths = []

        for category, score_data in scores.items():
            if isinstance(score_data, dict) and 'score' in score_data and 'evidence' in score_data:
                score = score_data['score']
                evidence = score_data['evidence']

                if category == 'open_source' and score > 0:
                    evidence = score_data.get('evidence', '').lower()
                    if any(phrase in evidence for phrase in [
                        'no evidence of significant open source contributions',
                        'no demonstrable open source activity',
                        'only personal github repositories',
                        'no contributions to other projects',
                        'lack of contributions to other projects'
                    ]):
                        pass
                    elif score >= 20:
                        strengths.append("Strong open source contributions with significant impact")
                    elif score >= 15:
                        strengths.append("Active participation in open source projects")
                    elif score >= 12:
                        strengths.append("Some open source involvement and community contributions")

                elif category == 'self_projects' and score > 0:
                    if score >= 20:
                        strengths.append("Impressive portfolio of self-initiated projects")
                    elif score >= 10:
                        strengths.append("Good variety of personal projects demonstrating technical skills")
                    elif score >= 5:
                        strengths.append("Some personal projects showing initiative")

                elif category == 'production' and score > 0:
                    if score >= 15:
                        strengths.append("Significant production experience at scale")
                    elif score >= 10:
                        strengths.append("Good production environment experience")
                    elif score >= 5:
                        strengths.append("Some production-level work experience")

                elif category == 'technical_skills' and score > 0:
                    if score >= 7:
                        strengths.append("Strong technical skills across multiple technologies")
                    elif score >= 5:
                        strengths.append("Good technical breadth and problem-solving abilities")
                    elif score >= 3:
                        strengths.append("Demonstrated technical skills in relevant areas")

        if not strengths:
            total_score = sum(score_data.get('score', 0) for score_data in scores.values() if isinstance(score_data, dict))
            if total_score > 0:
                strengths.append("Demonstrates technical capabilities and learning potential")
            else:
                strengths.append("Shows interest in software development")

        return strengths[:5]

    def _enforce_scoring_rules(self, scores: dict) -> dict:
        if 'open_source' in scores and isinstance(scores['open_source'], dict):
            evidence = scores['open_source'].get('evidence', '').lower()
            current_score = scores['open_source'].get('score', 0)

            if any(phrase in evidence for phrase in [
                'only personal github repositories',
                'no contributions to other projects',
                'no evidence of significant open source contributions',
                'only personal projects'
            ]):
                if current_score > 10:
                    logger.info(f"Enforcing scoring rule: reducing open source score from {current_score} to 10")
                    scores['open_source']['score'] = 10
                    scores['open_source']['evidence'] += " (Score capped at 10 due to only personal repositories)"

        return scores

    def _generate_improvements_from_scores(self, scores: dict) -> List[str]:
        improvements = []

        for category, score_data in scores.items():
            if isinstance(score_data, dict) and 'score' in score_data and 'evidence' in score_data:
                score = score_data['score']
                max_score = score_data.get('max', 0)
                evidence = score_data['evidence']

                percentage = (score / max_score * 100) if max_score > 0 else 0

                if category == 'open_source' and percentage < 70:
                    if percentage < 30:
                        improvements.append("Significantly increase open source contributions and community engagement")
                    elif percentage < 50:
                        improvements.append("Build more substantial open source presence and contributions")
                    else:
                        improvements.append("Enhance open source contributions and project impact")

                elif category == 'self_projects' and percentage < 70:
                    if percentage < 30:
                        improvements.append("Develop more complex and impactful personal projects")
                    elif percentage < 50:
                        improvements.append("Create projects with better documentation and user adoption")
                    else:
                        improvements.append("Enhance project complexity and real-world impact")

                elif category == 'production' and percentage < 70:
                    if percentage < 30:
                        improvements.append("Gain more production environment experience")
                    elif percentage < 50:
                        improvements.append("Seek opportunities for larger-scale production work")
                    else:
                        improvements.append("Expand production experience and responsibilities")

                elif category == 'technical_skills' and percentage < 70:
                    if percentage < 30:
                        improvements.append("Strengthen technical skills and problem-solving abilities")
                    elif percentage < 50:
                        improvements.append("Develop broader technical expertise")
                    else:
                        improvements.append("Enhance technical depth and competitive programming skills")

        if not improvements:
            total_score = sum(score_data.get('score', 0) for score_data in scores.values() if isinstance(score_data, dict))
            if total_score < 50:
                improvements.append("Focus on building a stronger technical portfolio")
            else:
                improvements.append("Continue developing technical skills and project impact")

        return improvements[:3]

    def _create_evaluation_result(self, data: EvaluationData) -> EvaluationResult:
        total_score = (
            data.scores.open_source.score +
            data.scores.self_projects.score +
            data.scores.production.score +
            data.scores.technical_skills.score
        )

        final_score = total_score + data.bonus_points.total - data.deductions.total

        category_breakdown = {
            'Open Source': (
                data.scores.open_source.score,
                data.scores.open_source.evidence
            ),
            'Self Projects': (
                data.scores.self_projects.score,
                data.scores.self_projects.evidence
            ),
            'Production': (
                data.scores.production.score,
                data.scores.production.evidence
            ),
            'Technical Skills': (
                data.scores.technical_skills.score,
                data.scores.technical_skills.evidence
            ),
            'Bonus Points': (
                data.bonus_points.total,
                data.bonus_points.breakdown
            )
        }

        if data.deductions.total > 0:
            category_breakdown['Deductions'] = (
                -data.deductions.total,
                data.deductions.reasons
            )

        scoring_explanation = self._create_scoring_explanation(data, total_score, final_score)

        return EvaluationResult(
            candidate_name=data.candidate_name,
            total_score=total_score,
            final_score=final_score,
            category_breakdown=category_breakdown,
            scoring_explanation=scoring_explanation,
            key_strengths=data.key_strengths,
            areas_for_improvement=data.areas_for_improvement
        )

    def _create_scoring_explanation(self, data: EvaluationData, total_score: float, final_score: float) -> str:
        explanation_parts = []

        explanation_parts.append(f"Base Score: {total_score}/100")
        explanation_parts.append(f"  - Open Source: {data.scores.open_source.score}/35")
        explanation_parts.append(f"  - Self Projects: {data.scores.self_projects.score}/30")
        explanation_parts.append(f"  - Production: {data.scores.production.score}/25")
        explanation_parts.append(f"  - Technical Skills: {data.scores.technical_skills.score}/10")

        if data.bonus_points.total > 0:
            explanation_parts.append(f"Bonus Points: +{data.bonus_points.total}")
            explanation_parts.append(f"  - {data.bonus_points.breakdown}")
        else:
            explanation_parts.append("Bonus Points: +0")

        if data.deductions.total > 0:
            explanation_parts.append(f"Deductions: -{data.deductions.total}")
            explanation_parts.append(f"  - {data.deductions.reasons}")
        else:
            explanation_parts.append("Deductions: -0")

        explanation_parts.append(f"Final Score: {total_score} + {data.bonus_points.total} - {data.deductions.total} = {final_score}")

        return "\n".join(explanation_parts)

    def evaluate_batch(self, resumes: List[Dict[str, str]]) -> List[EvaluationResult]:
        results = []
        for resume in resumes:
            try:
                result = self.evaluate_resume(resume['content'])
                results.append(result)
            except Exception as e:
                print(f"Failed to evaluate resume for {resume.get('name', 'Unknown')}: {str(e)}")

        return results

    def rank_candidates(self, results: List[EvaluationResult]) -> List[Tuple[str, float]]:
        rankings = [(r.candidate_name, r.final_score) for r in results]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
