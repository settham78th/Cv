import os
import logging
import requests
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
DEFAULT_MODEL = "mistralai/mistral-7b-instruct:free"

def create_system_prompt(task: str) -> str:
    """Create a specific system prompt based on the task."""
    base_prompt = "You are an expert HR professional and career advisor with extensive experience in CV/resume optimization."
    
    task_prompts = {
        "optimize": base_prompt + " Your task is to optimize CVs to maximize their impact while maintaining authenticity and professionalism.",
        "feedback": base_prompt + " You provide detailed, constructive feedback as an experienced recruiter would.",
        "cover_letter": base_prompt + " You specialize in creating compelling, personalized cover letters that highlight relevant experience.",
        "translate": "You are a professional translator specializing in CV/resume translation with perfect understanding of professional terminology.",
        "alternative_careers": base_prompt + " You excel at identifying transferable skills and suggesting alternative career paths.",
        "ats_check": base_prompt + " You are an expert in ATS systems and how they parse and score resumes.",
        "interview_questions": base_prompt + " You specialize in preparing candidates for job interviews with relevant, position-specific questions.",
        "market_trends": "You are a career market analyst with extensive knowledge of industry trends, skill demands, and salary ranges.",
    }
    
    return task_prompts.get(task, base_prompt)

def create_task_prompt(task: str, cv_text: str, job_description: str = "", additional_context: Dict[str, Any] = None) -> str:
    """Create a specific task prompt based on the operation type."""
    
    prompts = {
        "optimize": f"""Analyze and optimize the following CV for maximum impact while maintaining authenticity. Focus on:
1. Strong action verbs and quantifiable achievements
2. Clear, professional language
3. Relevant skills and experience highlighting
4. Proper formatting and structure
5. Keywords from the job description (if provided)

Job Description:
{job_description}

CV Text:
{cv_text}

Provide the optimized CV in a clear, well-structured format.""",

        "feedback": f"""Review the following CV as an experienced recruiter. Provide detailed feedback on:
1. Overall impression
2. Strengths and achievements
3. Areas for improvement
4. Alignment with job requirements
5. Specific recommendations

Job Description:
{job_description}

CV Text:
{cv_text}""",

        "cover_letter": f"""Create a compelling cover letter based on the CV and job description. Focus on:
1. Relevant experience and achievements
2. Specific examples demonstrating required skills
3. Company and role-specific customization
4. Professional tone and enthusiasm
5. Clear structure (introduction, body, conclusion)

Job Description:
{job_description}

CV Text:
{cv_text}""",

        "translate": f"""Translate the following CV to professional English, maintaining:
1. Industry-specific terminology
2. Professional formatting
3. Cultural adaptations where necessary
4. Consistent style and tone

CV Text:
{cv_text}""",

        "alternative_careers": f"""Analyze the following CV and suggest alternative career paths based on:
1. Transferable skills
2. Industry experience
3. Educational background
4. Career progression potential
5. Market demand

CV Text:
{cv_text}

For each suggested career path, explain:
- Why it's a good fit
- Required transitions or additional skills
- Potential career progression
- Market outlook""",

        "ats_check": f"""Analyze this CV's ATS compatibility against the job description. Evaluate:
1. Keyword matching and optimization
2. Formatting and structure
3. Essential qualifications alignment
4. Skills and experience relevance
5. Specific improvement recommendations

Job Description:
{job_description}

CV Text:
{cv_text}""",

        "interview_questions": f"""Based on this CV and job description, generate relevant interview questions:
1. Experience-based questions
2. Technical skill verification
3. Behavioral scenarios
4. Role-specific challenges
5. Career motivation and goals

Include suggested strong answers based on the CV content.

Job Description:
{job_description}

CV Text:
{cv_text}""",

        "market_trends": f"""Analyze market trends for {additional_context.get('job_title', '')} in the {additional_context.get('industry', 'general')} industry. Cover:
1. Current demand and future outlook
2. Required and emerging skills
3. Salary ranges and benefits
4. Industry-specific trends
5. Career progression opportunities"""
    }
    
    return prompts.get(task, "Please analyze the following text.")

def process_text_with_ai(text: str, task: str, job_description: str = "", additional_context: Optional[Dict[str, Any]] = None, model: str = DEFAULT_MODEL) -> Optional[str]:
    """Process text using OpenRouter API with improved prompts and error handling."""
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key is not set")
        return None
    
    if additional_context is None:
        additional_context = {}
    
    logger.info(f"Processing {task} request with OpenRouter AI")
    
    # Truncate text if too long
    max_chars = 12000
    if len(text) > max_chars:
        logger.warning(f"Text truncated from {len(text)} to {max_chars} characters")
        text = text[:max_chars] + "... [truncated]"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:5000",
        "X-Title": "CV Optimizer Pro"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": create_system_prompt(task)
            },
            {
                "role": "user",
                "content": create_task_prompt(task, text, job_description, additional_context)
            }
        ]
    }
    
    try:
        logger.info(f"Making API request for task: {task}")
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                if 'choices' in response_data and response_data['choices']:
                    ai_response = response_data['choices'][0]['message']['content']
                    logger.info(f"Successfully received AI response ({len(ai_response)} characters)")
                    return ai_response
                else:
                    logger.error(f"Unexpected response format: {response_data}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse API response as JSON: {response.text}")
        else:
            logger.error(f"API request failed with status code {response.status_code}: {response.text}")
    
    except requests.RequestException as e:
        logger.error(f"Request exception during API call: {str(e)}")
    
    return None

def optimize_cv(cv_text: str, job_description: str = "") -> str:
    """Optimize CV using AI."""
    result = process_text_with_ai(cv_text, "optimize", job_description)
    return result or "Failed to optimize CV. Please try again."

def optimize_cv_with_keywords(cv_text: str, job_description: str, keywords_data: Dict[str, Any]) -> str:
    """Optimize CV using AI with extracted keywords."""
    additional_context = {"keywords": keywords_data}
    result = process_text_with_ai(cv_text, "optimize", job_description, additional_context)
    return result or "Failed to optimize CV. Please try again."

def generate_recruiter_feedback(cv_text: str, job_description: str = "") -> str:
    """Generate detailed recruiter feedback."""
    result = process_text_with_ai(cv_text, "feedback", job_description)
    return result or "Failed to generate feedback. Please try again."

def generate_cover_letter(cv_text: str, job_description: str) -> str:
    """Generate a cover letter."""
    result = process_text_with_ai(cv_text, "cover_letter", job_description)
    return result or "Failed to generate cover letter. Please try again."

def translate_to_english(cv_text: str) -> str:
    """Translate CV to English."""
    result = process_text_with_ai(cv_text, "translate")
    return result or "Failed to translate CV. Please try again."

def suggest_alternative_careers(cv_text: str) -> str:
    """Suggest alternative career paths."""
    result = process_text_with_ai(cv_text, "alternative_careers")
    return result or "Failed to suggest alternative careers. Please try again."

def generate_multi_versions(cv_text: str, roles: list) -> str:
    """Generate multiple versions of CV for different roles."""
    versions = []
    for role in roles:
        result = process_text_with_ai(cv_text, "optimize", f"Role: {role}")
        versions.append(f"\n\n=== CV for {role} ===\n\n{result or 'Failed to generate this version.'}")
    return "\n".join(versions)

def ats_optimization_check(cv_text: str, job_description: str) -> str:
    """Check CV compatibility with ATS systems."""
    result = process_text_with_ai(cv_text, "ats_check", job_description)
    return result or "Failed to perform ATS check. Please try again."

def generate_interview_questions(cv_text: str, job_description: str) -> str:
    """Generate relevant interview questions."""
    result = process_text_with_ai(cv_text, "interview_questions", job_description)
    return result or "Failed to generate interview questions. Please try again."

def analyze_market_trends(job_title: str, industry: str = "") -> str:
    """Analyze market trends for a specific role."""
    additional_context = {"job_title": job_title, "industry": industry}
    result = process_text_with_ai("", "market_trends", additional_context=additional_context)
    return result or "Failed to analyze market trends. Please try again."

def analyze_job_url(url: str) -> str:
    """Extract and analyze job description from URL."""
    # This would typically use a web scraping library to extract the job description
    # For now, return an error message
    return "Job URL analysis is not implemented yet."

def extract_keywords_from_job(job_description: str) -> Dict[str, Any]:
    """Extract keywords from job description."""
    result = process_text_with_ai(job_description, "extract_keywords")
    if not result:
        return {}
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {}

def generate_keywords_html(keywords_data: Dict[str, Any]) -> str:
    """Generate HTML representation of keywords."""
    if not keywords_data:
        return "<p>No keywords found.</p>"
    
    html = []
    for category, keywords in keywords_data.items():
        html.append(f"<h4>{category}</h4>")
        html.append("<div class='mb-3'>")
        for keyword in keywords:
            html.append(f"<span class='badge bg-primary me-2 mb-2'>{keyword}</span>")
        html.append("</div>")
    
    return "\n".join(html)