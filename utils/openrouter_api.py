import os
import json
import logging
import requests
import urllib.parse
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Get API key from environment variables with fallback
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "mistralai/mistral-7b-instruct:free"  # Darmowy model Mistral

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://cv-optimizer-pro.repl.co/"  # Replace with your actual domain
}

def send_api_request(prompt, max_tokens=2000):
    """
    Send a request to the OpenRouter API
    """
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key not found")
        raise ValueError("OpenRouter API key not set in environment variables")

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert resume editor and career advisor. Always respond in the same language as the CV or job description provided by the user."},
            {"role": "user", "content": prompt}

def optimize_cv(cv_text, job_description):
    """
    Optimize CV based on job description
    """
    prompt = f"""
    Analyze the following CV and job description, then provide an optimized version of the CV:
    
    CV:
    {cv_text}
    
    Job Description:
    {job_description}
    
    Please optimize the CV to better match the job requirements while maintaining honesty and accuracy.
    """
    return send_api_request(prompt)

def optimize_cv_with_keywords(cv_text, job_description, keywords_data):
    """
    Optimize CV using extracted keywords
    """
    keywords_str = json.dumps(keywords_data, ensure_ascii=False, indent=2)
    prompt = f"""
    Optimize this CV using the job description and extracted keywords:
    
    CV:
    {cv_text}
    
    Job Description:
    {job_description}
    
    Keywords Data:
    {keywords_str}
    
    Please optimize the CV to incorporate these keywords naturally while maintaining honesty and accuracy.
    """
    return send_api_request(prompt)

def generate_recruiter_feedback(cv_text, job_description):
    """
    Generate recruiter-style feedback
    """
    prompt = f"""
    Review this CV for the following job description:
    
    CV:
    {cv_text}
    
    Job Description:
    {job_description}
    
    Provide detailed recruiter feedback including strengths and areas for improvement.
    """
    return send_api_request(prompt)

def generate_cover_letter(cv_text, job_description):
    """
    Generate a cover letter
    """
    prompt = f"""
    Create a cover letter based on this CV and job description:
    
    CV:
    {cv_text}
    
    Job Description:
    {job_description}
    
    Generate a professional cover letter that highlights relevant experience and skills.
    """
    return send_api_request(prompt)

def translate_to_english(cv_text):
    """
    Translate CV to English
    """
    prompt = f"""
    Translate this CV to professional English while maintaining formatting:
    
    {cv_text}
    """
    return send_api_request(prompt)

def suggest_alternative_careers(cv_text):
    """
    Suggest alternative career paths
    """
    prompt = f"""
    Based on this CV, suggest alternative career paths:
    
    {cv_text}
    
    Provide realistic career alternatives based on transferable skills and experience.
    """
    return send_api_request(prompt)

def generate_multi_versions(cv_text, roles):
    """
    Generate multiple versions of CV for different roles
    """
    roles_str = ", ".join(roles)
    prompt = f"""
    Create versions of this CV optimized for different roles:
    
    CV:
    {cv_text}
    
    Target Roles:
    {roles_str}
    
    Provide optimized versions for each role while maintaining core truthfulness.
    """
    return send_api_request(prompt)

def analyze_job_url(url):
    """
    Extract and analyze job description from URL
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Extract job description (this is a simple example, might need adjustment for specific sites)
        job_description = soup.get_text()
        return job_description.strip()
    except Exception as e:
        logger.error(f"Error extracting job description from URL: {str(e)}")
        raise Exception(f"Could not extract job description from URL: {str(e)}")

def ats_optimization_check(cv_text, job_description):
    """
    Check CV compatibility with ATS systems
    """
    prompt = f"""
    Analyze this CV for ATS compatibility:
    
    CV:
    {cv_text}
    
    Job Description:
    {job_description}
    
    Provide detailed feedback on ATS compatibility and suggestions for improvement.
    """
    return send_api_request(prompt)

def generate_interview_questions(cv_text, job_description):
    """
    Generate potential interview questions
    """
    prompt = f"""
    Generate relevant interview questions based on:
    
    CV:
    {cv_text}
    
    Job Description:
    {job_description}
    
    Provide a list of targeted interview questions and suggested answers.
    """
    return send_api_request(prompt)

def analyze_market_trends(job_title, industry):
    """
    Analyze market trends for the position
    """
    prompt = f"""
    Analyze market trends for:
    Job Title: {job_title}
    Industry: {industry}
    
    Provide insights on:
    1. Current market demand
    2. Required skills
    3. Salary ranges
    4. Future outlook
    """
    return send_api_request(prompt)

def extract_keywords_from_job(job_description):
    """
    Extract keywords from job description
    """
    prompt = f"""
    Analyze this job description and extract key terms:
    
    {job_description}
    
    Categorize and return keywords for:
    1. Technical Skills
    2. Soft Skills
    3. Experience Requirements
    4. Education Requirements
    5. Industry-Specific Terms
    """
    response = send_api_request(prompt)
    return json.loads(response) if isinstance(response, str) else response

def generate_keywords_html(keywords_data):
    """
    Generate HTML representation of keywords
    """
    html = ""
    for category, keywords in keywords_data.items():
        html += f"<h4>{category}</h4><div class='mb-3'>"
        for keyword in keywords:
            html += f"<span class='badge bg-primary me-2 mb-2'>{keyword}</span>"
        html += "</div>"
    return html

        ],
        "max_tokens": max_tokens
    }

    try:
        logger.debug(f"Sending request to OpenRouter API")
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload)
        response.raise_for_status()

        result = response.json()
        logger.debug(f"Received response from OpenRouter API")

        if 'choices' in result and len(result['choices']) > 0:
            return result['choices'][0]['message']['content']
        else:
            raise ValueError("Unexpected API response format")

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        raise Exception(f"Failed to communicate with OpenRouter API: {str(e)}")

    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Error parsing API response: {str(e)}")
        raise Exception(f"Failed to parse OpenRouter API response: {str(e)}")