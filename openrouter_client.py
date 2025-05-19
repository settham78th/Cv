import os
import logging
import requests
import json

logger = logging.getLogger(__name__)

# OpenRouter API configuration
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
DEFAULT_MODEL = "mistralai/mistral-7b-instruct:free"  # Darmowy model Mistral

def process_text_with_ai(text, prompt, model=DEFAULT_MODEL):
    """
    Process the extracted text using OpenRouter API.
    
    Args:
        text (str): The extracted text from the PDF
        prompt (str): The user's prompt/instructions for the AI
        model (str): The model to use for processing
        
    Returns:
        str: The AI-generated response or None if processing failed
    """
    if not OPENROUTER_API_KEY:
        logger.error("OpenRouter API key is not set")
        return None
    
    logger.info(f"Processing text with OpenRouter AI (length: {len(text)} characters)")
    
    # Truncate text if it's too long to avoid excessive token usage
    max_chars = 12000
    if len(text) > max_chars:
        logger.warning(f"Text is too long ({len(text)} chars), truncating to {max_chars} chars")
        text = text[:max_chars] + "... [text truncated due to length]"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:5000",  # Required by OpenRouter
        "X-Title": "PDF Text Processor"  # Optional but recommended by OpenRouter
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant tasked with processing and analyzing PDF text."
            },
            {
                "role": "user",
                "content": f"I have extracted the following text from a PDF document. Please {prompt}\n\nEXTRACTED TEXT:\n\n{text}"
            }
        ]
    }
    
    try:
        logger.info("Making API request to OpenRouter")
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
    
    logger.error("Failed to process text with AI")
    return None