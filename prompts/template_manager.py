"""
Template Manager for Section Extraction

This module provides functionality to load and render Jinja templates for
section-specific resume extraction prompts.
"""

import os
from typing import Dict, Optional
from jinja2 import Environment, FileSystemLoader, Template


class TemplateManager:
    """
    Manages Jinja templates for section-specific resume extraction.
    
    This class provides functionality to load and render templates for
    different resume sections (basics, work, education, skills, projects, awards).
    """
    
    def __init__(self, template_dir: str = "prompts/templates"):
        """
        Initialize the template manager.
        
        Args:
            template_dir (str): Directory containing Jinja templates
        """
        self.template_dir = template_dir
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self._templates: Dict[str, Template] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load all available templates."""
        template_files = {
            'basics': 'basics.jinja',
            'work': 'work.jinja',
            'education': 'education.jinja',
            'skills': 'skills.jinja',
            'projects': 'projects.jinja',
            'awards': 'awards.jinja',
            'system_message': 'system_message.jinja'
        }
        
        for section_name, filename in template_files.items():
            try:
                template_path = os.path.join(self.template_dir, filename)
                if os.path.exists(template_path):
                    self._templates[section_name] = self.env.get_template(filename)
                else:
                    print(f"⚠️ Template file not found: {template_path}")
            except Exception as e:
                print(f"❌ Error loading template {filename}: {e}")
    
    def get_available_sections(self) -> list:
        """
        Get list of available section names.
        
        Returns:
            list: List of available section names
        """
        return list(self._templates.keys())
    
    def render_template(self, section_name: str, **kwargs) -> Optional[str]:
        """
        Render a template for a specific section.
        
        Args:
            section_name (str): Name of the section (basics, work, education, etc.)
            **kwargs: Template variables (e.g., text_content)
            
        Returns:
            Optional[str]: Rendered template string, or None if template not found
        """
        if section_name not in self._templates:
            print(f"❌ Template not found for section: {section_name}")
            print(f"Available sections: {self.get_available_sections()}")
            return None
        
        try:
            template = self._templates[section_name]
            return template.render(**kwargs)
        except Exception as e:
            print(f"❌ Error rendering template for {section_name}: {e}")
            return None
    
    def render_basics_template(self, text_content: str) -> Optional[str]:
        """
        Render the basics template.
        
        Args:
            text_content (str): Resume text content
            
        Returns:
            Optional[str]: Rendered template string
        """
        return self.render_template('basics', text_content=text_content)
    
    def render_work_template(self, text_content: str) -> Optional[str]:
        """
        Render the work template.
        
        Args:
            text_content (str): Resume text content
            
        Returns:
            Optional[str]: Rendered template string
        """
        return self.render_template('work', text_content=text_content)
    
    def render_education_template(self, text_content: str) -> Optional[str]:
        """
        Render the education template.
        
        Args:
            text_content (str): Resume text content
            
        Returns:
            Optional[str]: Rendered template string
        """
        return self.render_template('education', text_content=text_content)
    
    def render_skills_template(self, text_content: str) -> Optional[str]:
        """
        Render the skills template.
        
        Args:
            text_content (str): Resume text content
            
        Returns:
            Optional[str]: Rendered template string
        """
        return self.render_template('skills', text_content=text_content)
    
    def render_projects_template(self, text_content: str) -> Optional[str]:
        """
        Render the projects template.
        
        Args:
            text_content (str): Resume text content
            
        Returns:
            Optional[str]: Rendered template string
        """
        return self.render_template('projects', text_content=text_content)
    
    def render_awards_template(self, text_content: str) -> Optional[str]:
        """
        Render the awards template.
        
        Args:
            text_content (str): Resume text content
            
        Returns:
            Optional[str]: Rendered template string
        """
        return self.render_template('awards', text_content=text_content)
    
    def render_system_message_template(self, section_name: str) -> Optional[str]:
        """
        Render the system message template.
        
        Args:
            section_name (str): Name of the section being extracted
            
        Returns:
            Optional[str]: Rendered template string
        """
        if 'system_message' not in self._templates:
            print(f"❌ Template not found for section: system_message")
            print(f"Available sections: {self.get_available_sections()}")
            return None
        
        try:
            template = self._templates['system_message']
            return template.render(section_name=section_name)
        except Exception as e:
            print(f"❌ Error rendering system message template: {e}")
            return None 