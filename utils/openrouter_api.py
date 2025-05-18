
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

def detect_seniority_level(cv_text, job_description):
    """
    Detect seniority level (junior, mid, senior) based on CV and job description
    """
    prompt = f"""
    TASK: Określ poziom seniority (junior, mid, senior) na podstawie CV i opisu stanowiska.
    
    Wskazówki do analizy:
    
    1. Sprawdź lata doświadczenia w CV
    2. Przeanalizuj poziom odpowiedzialności w poprzednich rolach
    3. Oceń wymagania z opisu stanowiska
    4. Porównaj umiejętności z CV z wymaganiami w opisie stanowiska
    
    Zwróć tylko jeden z poniższych poziomów:
    - "junior" - dla początkujących specjalistów z doświadczeniem 0-2 lata
    - "mid" - dla specjalistów z doświadczeniem 2-5 lat
    - "senior" - dla ekspertów z doświadczeniem 5+ lat
    
    CV:
    {cv_text[:2000]}...
    
    Opis stanowiska:
    {job_description[:2000]}...
    
    Odpowiedz tylko jednym słowem: junior, mid lub senior.
    """
    
    try:
        response = send_api_request(prompt, max_tokens=10)
        response = response.strip().lower()
        
        if response in ["junior", "mid", "senior"]:
            return response
        else:
            # Domyślnie zwróć mid-level jeśli odpowiedź jest nieprawidłowa
            logger.warning(f"Invalid seniority level detected: {response}. Using 'mid' as default.")
            return "mid"
    except Exception as e:
        logger.error(f"Error detecting seniority level: {str(e)}")
        return "mid"  # Domyślny poziom

def detect_industry(job_description):
    """
    Detect industry based on job description
    """
    prompt = f"""
    TASK: Określ branżę na podstawie opisu stanowiska.
    
    Możliwe branże:
    - "it" - technologia, programowanie, analiza danych, IT
    - "finance" - finanse, bankowość, księgowość, ubezpieczenia
    - "marketing" - marketing, reklama, PR, social media
    - "healthcare" - służba zdrowia, farmacja, medycyna
    - "hr" - HR, rekrutacja, zasoby ludzkie
    - "education" - edukacja, szkolnictwo, e-learning
    - "engineering" - inżynieria, produkcja, budownictwo
    - "legal" - prawo, usługi prawne
    - "creative" - kreatywna, design, sztuka, UX/UI
    - "general" - inna branża lub brak wyraźnej specjalizacji
    
    Opis stanowiska:
    {job_description[:2000]}...
    
    Odpowiedz tylko jednym słowem - kod branży.
    """
    
    try:
        response = send_api_request(prompt, max_tokens=10)
        response = response.strip().lower()
        
        valid_industries = ["it", "finance", "marketing", "healthcare", "hr", 
                           "education", "engineering", "legal", "creative", "general"]
        
        if response in valid_industries:
            return response
        else:
            # Domyślnie zwróć general jeśli odpowiedź jest nieprawidłowa
            logger.warning(f"Invalid industry detected: {response}. Using 'general' as default.")
            return "general"
    except Exception as e:
        logger.error(f"Error detecting industry: {str(e)}")
        return "general"  # Domyślna branża

def get_industry_specific_prompt(industry, seniority):
    """
    Get industry-specific prompt guidance
    """
    # Domyślne wskazówki dla ogólnej branży
    industry_guidance = """
    - Użyj uniwersalnego języka biznesowego
    - Podkreśl umiejętności interpersonalne i adaptacyjne
    - Skup się na osiągnięciach mierzalnych w różnych kontekstach
    - Podkreśl znajomość standardowych narzędzi biznesowych
    """
    
    # Branżowo-specyficzne wskazówki
    industry_prompts = {
        "it": """
    - Użyj technicznych terminów branżowych i nazw technologii
    - Wymień konkretne języki programowania, narzędzia, frameworki z określeniem poziomu biegłości
    - Podkreśl umiejętność rozwiązywania złożonych problemów technicznych
    - Uwzględnij metodyki wytwarzania oprogramowania (np. Agile, Scrum)
    - Wykorzystaj mierzalne wskaźniki techniczne (optymalizacja wydajności, redukcja błędów)
    - Uwzględnij projekty open source i repozytoria kodu (GitHub, GitLab)
    """,
        "finance": """
    - Zastosuj precyzyjny język finansowy i terminologię branżową
    - Podkreśl umiejętności analityczne i znajomość regulacji (np. MSSF, US GAAP)
    - Uwzględnij konkretne wyniki finansowe i optymalizacje kosztów w procentach
    - Wyeksponuj znajomość systemów finansowych i umiejętność analizy danych
    - Podkreśl dokładność i dbałość o szczegóły w kontekście finansowym
    """,
        "marketing": """
    - Użyj dynamicznego, kreatywnego języka z branżowym słownictwem marketingowym
    - Podaj konkretne wyniki kampanii (ROI, conversion rate, zasięg)
    - Wymień znajomość platform marketingowych i narzędzi analitycznych
    - Podkreśl umiejętności w zakresie content marketingu i mediów społecznościowych
    - Uwzględnij kreatywne projekty i case studies z mierzalnymi efektami
    """,
        "healthcare": """
    - Zastosuj profesjonalną terminologię medyczną
    - Podkreśl certyfikaty i uprawnienia branżowe
    - Wyeksponuj znajomość procedur medycznych i regulacji (np. RODO w kontekście danych medycznych)
    - Uwzględnij doświadczenie z konkretną aparaturą medyczną lub systemami opieki zdrowotnej
    - Podkreśl umiejętności interpersonalne w kontekście opieki nad pacjentem
    """,
        "hr": """
    - Zastosuj terminologię HR i zarządzania talentami
    - Podaj konkretne dane dotyczące rekrutacji, retencji i rozwoju pracowników
    - Podkreśl znajomość prawa pracy i systemów HR
    - Uwzględnij zrealizowane projekty rozwojowe i ich wpływ na organizację
    - Wyeksponuj umiejętności miękkie i komunikacyjne
    """,
        "education": """
    - Użyj terminologii edukacyjnej i pedagogicznej
    - Podkreśl certyfikaty nauczycielskie i metody edukacyjne
    - Uwzględnij opracowane materiały dydaktyczne i programy nauczania
    - Wyeksponuj mierzalne wyniki edukacyjne uczniów/studentów
    - Podkreśl umiejętności dydaktyczne i zarządzania klasą/grupą
    """,
        "engineering": """
    - Zastosuj precyzyjny język inżynieryjny i techniczny
    - Wymień konkretne projekty inżynieryjne z parametrami technicznymi
    - Podkreśl znajomość norm i standardów branżowych
    - Uwzględnij optymalizacje procesów i oszczędności materiałowe/czasowe
    - Wyeksponuj umiejętność rozwiązywania złożonych problemów technicznych
    """,
        "legal": """
    - Zastosuj precyzyjny język prawniczy i formalny styl
    - Podkreśl znajomość konkretnych aktów prawnych i orzecznictwa
    - Uwzględnij prowadzone sprawy/projekty z zachowaniem poufności
    - Wyeksponuj umiejętności analityczne i interpretacyjne
    - Podkreśl certyfikaty i uprawnienia prawnicze
    """,
        "creative": """
    - Użyj kreatywnego, dynamicznego języka
    - Uwzględnij portfolio projektów kreatywnych z konkretnymi efektami
    - Podkreśl znajomość narzędzi projektowych i technologii kreatywnych
    - Wyeksponuj umiejętność pracy w zespołach interdyscyplinarnych
    - Podkreśl nagrody i wyróżnienia w dziedzinach kreatywnych
    """
    }
    
    # Pobierz wskazówki dla konkretnej branży lub użyj domyślnych
    if industry in industry_prompts:
        industry_guidance = industry_prompts[industry]
    
    # Modyfikacje pod kątem seniority
    seniority_guidance = {
        "junior": """
    - Podkreśl zapał do nauki i szybkiego przyswajania wiedzy
    - Uwypuklij projekty szkolne/akademickie i ich praktyczne zastosowanie
    - Skup się na potencjale i umiejętnościach podstawowych
    - Pokaż gotowość do rozwoju pod mentorskim okiem
    """,
        "mid": """
    - Zbalansuj doświadczenie z potencjałem rozwojowym
    - Podkreśl samodzielnie zrealizowane projekty i ich efekty
    - Uwypuklij specjalizacje i konkretne obszary ekspertyzy
    - Pokaż umiejętność współpracy z różnymi interesariuszami
    """,
        "senior": """
    - Uwypuklij strategiczne myślenie i szerszą perspektywę biznesową
    - Podkreśl role przywódcze i mentorskie
    - Skup się na długofalowych efektach i transformacyjnych projektach
    - Pokaż umiejętność kierowania zespołami i zarządzania zasobami
    - Uwzględnij wpływ na działalność biznesową i KPI organizacji
    """
    }
    
    return industry_guidance + "\n" + seniority_guidance.get(seniority, seniority_guidance["mid"])

def get_measurable_achievements_prompt(seniority):
    """
    Get prompt to encourage adding measurable achievements based on seniority
    """
    prompts = {
        "junior": """
    Nawet dla juniora dodaj mierzalne osiągnięcia: 
    - Jeśli brak konkretnych liczb w CV, dodaj przybliżone wyniki: "przygotowałem około 10 analiz", "wsparłem X projektów"
    - Zamień ogólne stwierdzenia na konkretne: "nauczyłem się X technologii w ciągu 3 miesięcy"
    - Uwzględnij efekty edukacyjne: "ukończyłem studia z wynikiem X% / w czołówce Y% studentów"
    - Dodaj wyniki projektów studenckich/hobbyistycznych z konkretnymi liczbami
    """,
        "mid": """
    Wzbogać CV o konkretne, mierzalne wyniki:
    - Dodaj procenty poprawy procesów: "zwiększyłem wydajność o X%", "skróciłem czas realizacji o Y dni"
    - Uwzględnij konkretne wskaźniki: "przeprowadziłem X kampanii", "wdrożyłem Y funkcjonalności"
    - Zamień ogólniki na liczby: "zarządzałem 5-osobowym zespołem", "pozyskałem X nowych klientów"
    - Dodaj skalę projektów: "projekt o budżecie X zł", "system dla Y użytkowników"
    """,
        "senior": """
    Umieść strategiczne, biznesowe mierzalne osiągnięcia:
    - Dodaj wskaźniki finansowe: "zwiększyłem przychody o X%", "zredukowałem koszty o Y zł"
    - Uwzględnij wpływ na organizację: "wdrożyłem strategię, która zwiększyła rentowność o X%"
    - Podkreśl efekty przywództwa: "kierowałem zespołem X osób, osiągając Y% wzrostu produktywności"
    - Zamień każde ogólnikowe osiągnięcie na konkretne z liczbami, procentami i skalą czasową
    - Uwzględnij wyniki transformacji: "przeprowadziłem restrukturyzację działu X, co przyniosło Y oszczędności"
    """
    }
    
    return prompts.get(seniority, prompts["mid"])

def get_structural_quality_control_prompt(seniority, industry):
    """
    Get structural quality control prompt based on seniority and industry
    """
    base_prompt = """
    Zapewnij optymalną strukturę CV:
    - Akapity nie dłuższe niż 3-4 linijki tekstu
    - Każde doświadczenie zawodowe opisane w 3-5 punktach
    - Sekcja umiejętności podzielona na kategorie
    - Zachowaj spójny format dla dat i lokalizacji
    - Stosuj nagłówki w standardzie ATS
    """
    
    industry_specific = {
        "it": """
    - Dodaj sekcję umiejętności technicznych na początku, kategoryzując je
    - Dla każdej technologii określ poziom zaawansowania (%)
    - Używaj wypunktowań dla osiągnięć technicznych (4-6 punktów na rolę)
    - Skróć historię zawodową do najważniejszych technologicznie stanowisk
    """,
        "finance": """
    - Użyj precyzyjnych nagłówków sekcji (np. "Doświadczenie w księgowości zarządczej")
    - Każdy punkt osiągnięć powinien mieć aspekt ilościowy
    - Struktura punktów: działanie, sposób, rezultat, skala
    - Zachowaj formalny układ bez elementów kreatywnych
    """,
        "marketing": """
    - Użyj kreatywnych, ale jasnych nagłówków sekcji
    - Każde doświadczenie zawodowe opisz w 4-6 punktach
    - Zrównoważ aspekty kreatywne i analityczne w punktach
    - Dodaj sekcję z przykładami kampanii/projektów
    """,
        "creative": """
    - Zastosuj przejrzysty układ podkreślający portfolio
    - Punkty osiągnięć skup na efekcie i procesie kreatywnym
    - Używaj dynamicznych czasowników na początku punktów
    - Zrównoważ techniczne aspekty z kreatywnymi
    """
    }
    
    language_style = {
        "junior": """
    - Użyj prostego, bezpośredniego języka
    - Stosuj podstawową terminologię branżową
    - Unikaj zaawansowanego słownictwa
    - Podkreślaj entuzjazm i potencjał
    """,
        "mid": """
    - Zbalansuj profesjonalny język z przystępnością
    - Stosuj branżowe terminy w kontekście
    - Unikaj zbyt ogólnikowych stwierdzeń
    - Zachowaj spójność stylu w całym dokumencie
    """,
        "senior": """
    - Stosuj zaawansowany język biznesowy i branżowy
    - Używaj precyzyjnych terminów strategicznych
    - Podkreślaj aspekty przywódcze w stylu komunikacji
    - Zachowaj profesjonalny, pewny siebie ton
    """
    }
    
    industry_guidance = industry_specific.get(industry, "")
    style_guidance = language_style.get(seniority, language_style["mid"])
    
    return base_prompt + "\n" + industry_guidance + "\n" + style_guidance

def optimize_cv_with_keywords(cv_text, job_description, keywords_data=None):
    """
    Create an optimized version of CV using advanced AI processing with focus on specific keywords
    """
    # Jeśli nie podano słów kluczowych, spróbuj je wygenerować
    if keywords_data is None:
        try:
            keywords_data = extract_keywords_from_job(job_description)
        except Exception as e:
            logger.error(f"Failed to extract keywords for CV optimization: {str(e)}")
            keywords_data = {}
    
    # Wykryj poziom doświadczenia i branżę
    try:
        seniority = detect_seniority_level(cv_text, job_description)
        logger.info(f"Detected seniority level: {seniority}")
        
        industry = detect_industry(job_description)
        logger.info(f"Detected industry: {industry}")
    except Exception as e:
        logger.error(f"Error detecting context: {str(e)}")
        seniority = "mid"  # Domyślny poziom
        industry = "general"  # Domyślna branża
    
    # Pobierz specyficzne wytyczne dla branży i poziomu
    industry_prompt = get_industry_specific_prompt(industry, seniority)
    achievements_prompt = get_measurable_achievements_prompt(seniority)
    structural_prompt = get_structural_quality_control_prompt(seniority, industry)
    
    # Przygotuj dodatkowe wytyczne na podstawie słów kluczowych
    keyword_instructions = ""
    
    if keywords_data and isinstance(keywords_data, dict):
        keyword_instructions = "KLUCZOWE SŁOWA, KTÓRE NALEŻY UWZGLĘDNIĆ:\n\n"
        
        # Dodaj wysokopriorytetowe słowa kluczowe
        high_priority_keywords = []
        
        for category, words in keywords_data.items():
            category_name = category.replace("_", " ").title()
            for word in words:
                if isinstance(word, dict) and "slowo" in word and "waga" in word:
                    if word["waga"] >= 4:  # Wysoki priorytet
                        high_priority_keywords.append(f"{word['slowo']} ({category_name})")
        
        if high_priority_keywords:
            keyword_instructions += "Najważniejsze słowa kluczowe (koniecznie uwzględnij):\n"
            for kw in high_priority_keywords:
                keyword_instructions += f"- {kw}\n"
            keyword_instructions += "\n"
        
        # Dodaj kategoryzowane słowa
        for category, words in keywords_data.items():
            if words:
                category_name = category.replace("_", " ").title()
                keyword_instructions += f"{category_name}:\n"
                
                for word in words:
                    if isinstance(word, dict) and "slowo" in word:
                        keyword_instructions += f"- {word['slowo']}\n"
                
                keyword_instructions += "\n"
    
    prompt = f"""
    TASK: Stwórz całkowicie nową, spersonalizowaną wersję CV precyzyjnie dopasowaną do wymagań stanowiska.
    
    Wykryty poziom doświadczenia: {seniority.upper()}
    Wykryta branża: {industry.upper()}
    
    Kluczowe wytyczne optymalizacji:
    1. Głęboka analiza i transformacja doświadczenia:
       - Przeprowadź szczegółową analizę każdego stanowiska pod kątem wymagań nowej roli
       - Zidentyfikuj i wyeksponuj transferowalne umiejętności
       - Dodaj wymierzalne rezultaty i osiągnięcia (%, liczby, skala projektów)
       - Użyj profesjonalnego, branżowego słownictwa charakterystycznego dla danego sektora
       - Stwórz nowe opisy stanowisk wykorzystując słowa kluczowe z ogłoszenia
    
    2. Zaawansowana personalizacja:
       - Dopasuj tone of voice do kultury firmy i branży
       - Uwzględnij specyficzne technologie i metodologie wymienione w ogłoszeniu
       - Dodaj sekcję highlight'ów dopasowaną do priorytetowych wymagań
       - Stwórz spersonalizowane podsumowanie zawodowe podkreślające najważniejsze atuty
    
    3. Optymalizacja umiejętności i kompetencji:
       - Podziel umiejętności na kategorie: techniczne, miękkie, branżowe
       - Określ poziom zaawansowania w skali 1-5 dla kluczowych kompetencji
       - Dodaj konkretne przykłady zastosowania każdej kluczowej umiejętności
       - Uwzględnij certyfikaty i szkolenia istotne dla stanowiska
    
    4. Wytyczne branżowo-specyficzne:
    {industry_prompt}
    
    5. Wytyczne odnośnie mierzalnych osiągnięć:
    {achievements_prompt}
    
    6. Wytyczne dotyczące struktury i jakości:
    {structural_prompt}
    
    {keyword_instructions}
    
    WAŻNE ZASADY:
    - Zachowaj pełną spójność z prawdą zawartą w oryginalnym CV
    - Każda sekcja musi być napisana od nowa z fokusem na nowe stanowisko
    - Używaj aktywnych czasowników i konkretnych przykładów
    - Odpowiedz w tym samym języku co oryginalne CV
    - KONIECZNIE uwzględnij najważniejsze słowa kluczowe wymienione powyżej
    
    DANE:
    
    Opis stanowiska:
    {job_description}
    
    Oryginalne CV:
    {cv_text}
    
    Zwróć tylko zoptymalizowane CV w formacie tekstowym, bez dodatkowych komentarzy.
    """
    
    return send_api_request(prompt, max_tokens=2500)

def optimize_cv(cv_text, job_description):
    """
    Create an optimized version of CV using advanced AI processing
    """
    # Dla zachowania kompatybilności, wywołaj nową funkcję
    return optimize_cv_with_keywords(cv_text, job_description)

def generate_recruiter_feedback(cv_text, job_description=""):
    """
    Generate feedback on a CV as if from an AI recruiter
    """
    context = ""
    if job_description:
        context = f"Job description for context:\n{job_description}"
        
    prompt = f"""
    TASK: You are an experienced professional recruiter. Review this CV and provide detailed, actionable feedback.
    
    Include:
    1. Overall impression
    2. Strengths and weaknesses
    3. Formatting and structure assessment
    4. Content quality evaluation
    5. ATS compatibility
    6. Specific improvement suggestions
    7. Rating out of 10
    
    IMPORTANT: Respond in the same language as the CV. If the CV is in Polish, respond in Polish. If the CV is in English, respond in English.
    
    {context}
    
    CV:
    {cv_text}
    
    Provide detailed recruiter feedback. Be honest but constructive.
    """
    
    return send_api_request(prompt, max_tokens=2000)

def generate_cover_letter(cv_text, job_description):
    """
    Generate a cover letter based on a CV and job description
    """
    prompt = f"""
    TASK: Create a personalized cover letter based on this CV and job description.
    
    The cover letter should:
    - Be professionally formatted
    - Highlight relevant skills and experiences from the CV
    - Connect the candidate's background to the job requirements
    - Include a compelling introduction and conclusion
    - Be approximately 300-400 words
    
    IMPORTANT: Respond in the same language as the CV. If the CV is in Polish, respond in Polish. If the CV is in English, respond in English.
    
    Job description:
    {job_description}
    
    CV:
    {cv_text}
    
    Return only the cover letter in plain text format.
    """
    
    return send_api_request(prompt, max_tokens=2000)

def translate_to_english(cv_text):
    """
    Translate a CV to English while preserving professional terminology
    """
    prompt = f"""
    TASK: Translate this CV to professional English.
    
    Important:
    - Maintain all professional terminology
    - Preserve the original structure and formatting
    - Ensure proper translation of industry-specific terms
    - Keep names of companies and products unchanged
    - Make sure the translation sounds natural and professional in English
    
    Original CV:
    {cv_text}
    
    Return only the translated CV in plain text format.
    """
    
    return send_api_request(prompt, max_tokens=2500)

def suggest_alternative_careers(cv_text):
    """
    Suggest alternative career paths based on the skills in a CV
    """
    prompt = f"""
    TASK: Analyze this CV and suggest alternative career paths based on the skills and experience.
    
    For each suggested career path include:
    1. Job title/role
    2. Why it's a good fit based on existing skills
    3. What additional skills might be needed
    4. Potential industries or companies to target
    5. Estimated effort to transition (low/medium/high)
    
    IMPORTANT: Respond in the same language as the CV. If the CV is in Polish, respond in Polish. If the CV is in English, respond in English.
    
    CV:
    {cv_text}
    
    Provide a detailed analysis with specific, actionable recommendations.
    """
    
    return send_api_request(prompt, max_tokens=2000)

def generate_multi_versions(cv_text, roles):
    """
    Generate multiple versions of a CV tailored to different roles
    """
    roles_text = "\n".join([f"- {role}" for role in roles])
    
    prompt = f"""
    TASK: Create tailored versions of this CV for different roles.
    
    Roles to create CV versions for:
    {roles_text}
    
    For each role:
    1. Highlight relevant skills and experiences
    2. Customize the professional summary
    3. Adjust the emphasis of achievements
    4. Remove or downplay irrelevant information
    
    IMPORTANT: Respond in the same language as the CV. If the CV is in Polish, respond in Polish. If the CV is in English, respond in English.
    
    Original CV:
    {cv_text}
    
    Return each version clearly separated with a heading indicating the role.
    """
    
    return send_api_request(prompt, max_tokens=3000)

def analyze_job_url(url):
    """
    Extract job description from a URL with improved handling for popular job sites
    """
    try:
        logger.debug(f"Analyzing job URL: {url}")
        
        # Validate URL
        parsed_url = urllib.parse.urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        # Fetch the page
        response = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        response.raise_for_status()
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try to find the job description
        job_text = ""
        domain = parsed_url.netloc.lower()
        
        # Enhanced site-specific extraction for popular job boards
        if 'linkedin.com' in domain:
            containers = soup.select('.description__text, .show-more-less-html, .jobs-description__content')
            if containers:
                job_text = containers[0].get_text(separator='\n', strip=True)
                
        elif 'indeed.com' in domain:
            container = soup.select_one('#jobDescriptionText')
            if container:
                job_text = container.get_text(separator='\n', strip=True)
                
        elif 'pracuj.pl' in domain:
            containers = soup.select('[data-test="section-benefit-expectations-text"], [data-test="section-description-text"]')
            if containers:
                job_text = '\n'.join([c.get_text(separator='\n', strip=True) for c in containers])
                
        elif 'olx.pl' in domain or 'praca.pl' in domain:
            containers = soup.select('.offer-description, .offer-content, .description')
            if containers:
                job_text = containers[0].get_text(separator='\n', strip=True)
        
        # If no specific site pattern matched, use generic approach
        if not job_text:
            # Look for common containers for job descriptions
            potential_containers = soup.select('.job-description, .description, .details, article, .job-content, [class*=job], [class*=description], [class*=offer]')
            if potential_containers:
                # Get the longest container text as it's likely the main description
                for container in potential_containers:
                    container_text = container.get_text(separator='\n', strip=True)
                    if len(container_text) > len(job_text):
                        job_text = container_text
            
            # If still no container found, get the body text
            if not job_text and soup.body:
                # Remove navigation, header, footer and scripts
                for tag in soup.select('nav, header, footer, script, style, iframe'):
                    tag.decompose()
                
                job_text = soup.body.get_text(separator='\n', strip=True)
                
                # If body text is very long, try to extract the most relevant part
                if len(job_text) > 10000:
                    paragraphs = job_text.split('\n')
                    # Look for paragraphs with keywords likely to be in job descriptions
                    keywords = ['requirements', 'responsibilities', 'qualifications', 'skills', 'experience', 'about the job', 
                                'wymagania', 'obowiązki', 'kwalifikacje', 'umiejętności', 'doświadczenie', 'o pracy']
                    
                    relevant_paragraphs = []
                    found_relevant = False
                    
                    for paragraph in paragraphs:
                        if any(keyword.lower() in paragraph.lower() for keyword in keywords):
                            found_relevant = True
                        if found_relevant and len(paragraph.strip()) > 50:  # Only include substantive paragraphs
                            relevant_paragraphs.append(paragraph)
                    
                    if relevant_paragraphs:
                        job_text = '\n'.join(relevant_paragraphs)
        
        # Clean up the text - remove excessive whitespace but preserve paragraph breaks
        job_text = '\n'.join([' '.join(line.split()) for line in job_text.split('\n') if line.strip()])
        
        if not job_text:
            raise ValueError("Could not extract job description from the URL")
        
        logger.debug(f"Successfully extracted job description from URL")
        
        # If the text is too long, summarize it using the AI
        if len(job_text) > 4000:
            logger.debug(f"Job description is long ({len(job_text)} chars), summarizing with AI")
            job_text = summarize_job_description(job_text)
        
        return job_text
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching job URL: {str(e)}")
        raise Exception(f"Failed to fetch job posting from URL: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error analyzing job URL: {str(e)}")
        raise Exception(f"Failed to analyze job posting: {str(e)}")

def summarize_job_description(job_text):
    """
    Summarize a long job description using the AI
    """
    prompt = f"""
    TASK: Extract and summarize the key information from this job posting.
    
    Include:
    1. Job title and company (if mentioned)
    2. Required skills and qualifications
    3. Responsibilities and duties
    4. Preferred experience
    5. Any other important details (benefits, location, etc.)
    6. TOP 5 keywords that are critically important for this position
    
    IMPORTANT: Detect the language of the job posting and respond in that same language.
    If the job posting is in Polish, respond in Polish.
    If the job posting is in English, respond in English.
    
    Job posting text:
    {job_text[:4000]}...
    
    Provide a concise but comprehensive summary of this job posting, focusing on information relevant for CV optimization.
    Format the TOP 5 keywords as a separate section at the end labeled "KLUCZOWE SŁOWA:" (in Polish) or "KEY KEYWORDS:" (in English).
    """
    
    return send_api_request(prompt, max_tokens=1500)

def extract_keywords_from_job(job_description):
    """
    Extract key keywords from a job description for CV optimization
    """
    prompt = f"""
    TASK: Przeanalizuj dokładnie ten opis stanowiska i wyodrębnij kluczowe słowa i umiejętności.
    
    Pogrupuj słowa kluczowe w następujące kategorie:
    1. Umiejętności techniczne (8-12 słów)
    2. Wymagane doświadczenie (3-5 słów)
    3. Cechy osobowości (3-5 słów)
    4. Kluczowe obowiązki (4-6 słów)
    5. Branżowe terminy (4-6 słów)
    
    Każde słowo kluczowe powinno być pojedynczym wyrazem lub krótkim wyrażeniem (max 3 słowa).
    
    Dodaj ocenę ważności (1-5) dla każdego słowa kluczowego.
    
    Opis stanowiska:
    {job_description}
    
    Zwróć wyniki w formacie JSON DOKŁADNIE w takiej strukturze (bez zmian w nazwach pól):
    {{
        "umiejetnosci_techniczne": [
            {{"slowo": "Python", "waga": 5}},
            {{"slowo": "SQL", "waga": 4}}
        ],
        "wymagane_doswiadczenie": [
            {{"slowo": "5 lat w IT", "waga": 5}}
        ],
        "cechy_osobowosci": [
            {{"slowo": "Komunikatywność", "waga": 3}}
        ],
        "kluczowe_obowiazki": [
            {{"slowo": "Tworzenie raportów", "waga": 4}}
        ],
        "branzowe_terminy": [
            {{"slowo": "API", "waga": 4}}
        ]
    }}
    
    WAŻNE: Odpowiedz wyłącznie w formacie JSON bez dodatkowych komentarzy. Upewnij się, że Twoja odpowiedź może być bezpośrednio przekonwertowana do obiektu przez json.loads() bez żadnych modyfikacji.
    """
    
    response = send_api_request(prompt, max_tokens=1500)
    
    try:
        # Próba usunięcia znaczników kodu z odpowiedzi, jeśli istnieją
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()
        
        # Usunięcie ewentualnych komentarzy przed lub po JSON
        import re
        json_match = re.search(r'(\{.*\})', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
            
        # Konwersja tekstu na obiekt JSON
        keywords_data = json.loads(response)
        
        # Sprawdzenie czy struktura danych jest poprawna
        required_keys = ["umiejetnosci_techniczne", "wymagane_doswiadczenie", "cechy_osobowosci", 
                        "kluczowe_obowiazki", "branzowe_terminy"]
        
        for key in required_keys:
            if key not in keywords_data:
                # Jeśli brakuje któregoś klucza, dodaj pusty
                keywords_data[key] = []
                
            # Upewnij się, że wszystkie elementy mają wymagane pola
            fixed_items = []
            for item in keywords_data[key]:
                if isinstance(item, dict) and "slowo" in item and "waga" in item:
                    fixed_items.append(item)
                elif isinstance(item, dict) and "slowo" in item:
                    # Jeśli brakuje wagi, dodaj domyślną
                    item["waga"] = 3
                    fixed_items.append(item)
                elif isinstance(item, str):
                    # Jeśli to po prostu string, zamień na odpowiedni format
                    fixed_items.append({"slowo": item, "waga": 3})
                    
            keywords_data[key] = fixed_items
                
        return keywords_data
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"Error parsing keywords response: {str(e)}")
        # W przypadku błędu, zwróć podstawową strukturę z komunikatem błędu
        return {
            "umiejetnosci_techniczne": [{"slowo": "Błąd analizy - spróbuj ponownie", "waga": 3}],
            "wymagane_doswiadczenie": [{"slowo": "Wystąpił problem podczas analizy", "waga": 3}],
            "cechy_osobowosci": [{"slowo": "Spróbuj z krótszym opisem", "waga": 3}],
            "kluczowe_obowiazki": [{"slowo": "Nie udało się wyodrębnić", "waga": 3}],
            "branzowe_terminy": [{"slowo": f"Błąd: {str(e)[:30]}", "waga": 3}]
        }

def generate_keywords_html(keywords_data):
    """
    Generate HTML visualization for keywords from job description
    """
    html = """
    <div class="keywords-container">
        <h3>Słowa kluczowe znalezione w opisie stanowiska</h3>
    """
    
    # Funkcja pomocnicza do generowania kolorów na podstawie wagi
    def get_weight_color(weight):
        if weight >= 5:
            return "danger"  # Czerwony dla najważniejszych
        elif weight >= 4:
            return "warning"  # Pomarańczowy/żółty
        elif weight >= 3:
            return "success"  # Zielony
        elif weight >= 2:
            return "info"     # Niebieski
        else:
            return "secondary"  # Szary dla najmniej ważnych
    
    # Generowanie HTML dla każdej kategorii
    for category, keywords in keywords_data.items():
        category_name = category.replace("_", " ").title()
        html += f'<div class="keyword-category mb-3"><h4>{category_name}</h4><div class="d-flex flex-wrap">'
        
        for keyword in keywords:
            # Sprawdzenie czy keyword jest obiektem czy stringiem
            if isinstance(keyword, dict):
                weight = keyword.get("waga", 3)
                word = keyword.get("slowo", "")
            else:
                # Jeśli to string, użyj go jako słowa z domyślną wagą
                weight = 3
                word = str(keyword)
                
            color = get_weight_color(weight)
            
            html += f'<span class="badge bg-{color} m-1 p-2" data-weight="{weight}">{word}</span>'
        
        html += '</div></div>'
    
    html += """
    <div class="mt-3">
        <small class="text-muted">* Kolor wskazuje na wagę słowa kluczowego - ciemniejsze kolory oznaczają większe znaczenie</small>
    </div>
    </div>
    """
    
    return html

def analyze_market_trends(job_title, industry=""):
    """
    Analyze market trends and suggest popular skills/keywords for a specific job or industry
    """
    prompt = f"""
    TASK: Przeprowadź analizę trendów rynkowych dla pozycji: {job_title} w branży: {industry if industry else "wszystkich branżach"}.
    
    Przygotuj następujące informacje:
    1. TOP 10 najbardziej poszukiwanych umiejętności technicznych dla tej pozycji w 2025 roku
    2. TOP 5 umiejętności miękkich cenionych przez pracodawców
    3. Technologie/narzędzia wschodząće, które warto wymienić w CV
    4. 3 najważniejsze certyfikaty lub szkolenia zwiększające wartość kandydata
    5. Trendy płacowe - przedziały wynagrodzeń 
    
    Format odpowiedzi powinien być zwięzły, przejrzysty i łatwy do odczytania przez osobę szukającą pracy.
    Jeśli nazwa stanowiska jest w języku angielskim, odpowiedz po angielsku. Jeśli w języku polskim - odpowiedz po polsku.
    """
    
    return send_api_request(prompt, max_tokens=1500)

def ats_optimization_check(cv_text, job_description=""):
    """
    Check CV against ATS (Applicant Tracking System) and provide suggestions for improvement
    """
    context = ""
    if job_description:
        context = f"Ogłoszenie o pracę dla odniesienia:\n{job_description[:2000]}"
        
    prompt = f"""
    TASK: Sprawdź to CV pod kątem kompatybilności z systemami ATS (Applicant Tracking System) i oceń jego skuteczność.
    
    Przeprowadź następujące analizy:
    1. Wykryj potencjalne problemy z formatowaniem, które mogą utrudnić odczyt przez systemy ATS
    2. Sprawdź gęstość słów kluczowych i trafność ich wykorzystania
    3. Oceń strukturę CV i zaproponuj ulepszenia dla lepszej czytelności maszynowej
    4. Zidentyfikuj brakujące sekcje lub informacje, które są często wymagane przez ATS
    5. Oceń ogólną skuteczność CV w systemach ATS w skali 1-10
    
    {context}
    
    CV do analizy:
    {cv_text}
    
    Odpowiedz w tym samym języku co CV. Jeśli CV jest po polsku, odpowiedz po polsku.
    Format odpowiedzi powinien być przejrzysty, z wyraźnie oznaczonymi sekcjami i konkretnymi sugestiami usprawnień.
    """
    
    return send_api_request(prompt, max_tokens=1800)

def generate_interview_questions(cv_text, job_description=""):
    """
    Generate likely interview questions based on CV and job description
    """
    context = ""
    if job_description:
        context = f"Uwzględnij poniższe ogłoszenie o pracę przy tworzeniu pytań:\n{job_description[:2000]}"
        
    prompt = f"""
    TASK: Wygeneruj zestaw potencjalnych pytań rekrutacyjnych, które kandydat może otrzymać podczas rozmowy kwalifikacyjnej.
    
    Pytania powinny być:
    1. Specyficzne dla doświadczenia i umiejętności kandydata wymienionych w CV
    2. Dopasowane do stanowiska (jeśli podano opis stanowiska)
    3. Zróżnicowane - połączenie pytań technicznych, behawioralnych i sytuacyjnych
    4. Realistyczne i często zadawane przez rekruterów
    
    Uwzględnij po co najmniej 3 pytania z każdej kategorii:
    - Pytania o doświadczenie zawodowe
    - Pytania techniczne/o umiejętności
    - Pytania behawioralne
    - Pytania sytuacyjne
    - Pytania o motywację i dopasowanie do firmy/stanowiska
    
    {context}
    
    CV:
    {cv_text}
    
    Odpowiedz w tym samym języku co CV. Jeśli CV jest po polsku, odpowiedz po polsku.
    Dodatkowo, do każdego pytania dodaj krótką wskazówkę, jak można by na nie odpowiedzieć w oparciu o informacje z CV.
    """
    
    return send_api_request(prompt, max_tokens=2000)
