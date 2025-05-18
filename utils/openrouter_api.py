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

def detect_job_type(job_description):
    """
    Detect job type (physical, technical, office) based on job description
    """
    prompt = f"""
    TASK: Określ typ pracy opisanej w ogłoszeniu o pracę.

    Możliwe typy pracy:
    - "physical" - praca fizyczna (np. kierowca, magazynier, pracownik produkcji)
    - "technical" - praca techniczna (np. mechanik, elektryk, technik)
    - "office" - praca biurowa (np. administrator, asystent, koordynator)
    - "professional" - praca specjalistyczna (np. lekarz, prawnik, nauczyciel)
    - "creative" - praca kreatywna (np. grafik, projektant, artysta)
    - "it" - praca w IT (np. programista, administrator sieci, analityk danych)

    Opis stanowiska:
    {job_description[:2000]}...

    Odpowiedz tylko jednym słowem - kod typu pracy.
    """

    try:
        response = send_api_request(prompt, max_tokens=10)
        response = response.strip().lower()

        valid_job_types = ["physical", "technical", "office", "professional", "creative", "it"]

        if response in valid_job_types:
            return response
        else:
            # Domyślnie zwróć office jeśli odpowiedź jest nieprawidłowa
            logger.warning(f"Invalid job type detected: {response}. Using 'office' as default.")
            return "office"
    except Exception as e:
        logger.error(f"Error detecting job type: {str(e)}")
        return "office"  # Domyślny typ pracy

def detect_specific_role(job_description):
    """
    Detect specific job role based on job description
    """
    prompt = f"""
    TASK: Określ konkretną rolę zawodową opisaną w ogłoszeniu o pracę.

    Wybierz jedną konkretną rolę, która najlepiej pasuje do opisu, na przykład:
    - kierowca
    - magazynier
    - sprzedawca
    - księgowy
    - programista
    - nauczyciel
    - lekarz
    - grafik
    - mechanik
    - inżynier

    Opis stanowiska:
    {job_description[:2000]}...

    Odpowiedz tylko jednym słowem - nazwa konkretnej roli zawodowej, bez żadnych dodatkowych słów.
    """

    try:
        response = send_api_request(prompt, max_tokens=10)
        return response.strip().lower()
    except Exception as e:
        logger.error(f"Error detecting specific role: {str(e)}")
        return "specjalista"  # Domyślna rola

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
    - "transport" - transport, logistyka, spedycja
    - "retail" - handel detaliczny, sprzedaż, obsługa klienta
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
                           "education", "engineering", "transport", "retail",
                           "legal", "creative", "general"]

        if response in valid_industries:
            return response
        else:
            # Domyślnie zwróć general jeśli odpowiedź jest nieprawidłowa
            logger.warning(f"Invalid industry detected: {response}. Using 'general' as default.")
            return "general"
    except Exception as e:
        logger.error(f"Error detecting industry: {str(e)}")
        return "general"  # Domyślna branża

def get_role_specific_competencies(role):
    """
    Get role-specific competencies, certifications and typical achievements
    """
    role_competencies = {
        "kierowca": {
            "certifications": [
                "Prawo jazdy kategorii B/C/C+E/D",
                "Karta kierowcy",
                "Świadectwo kwalifikacji zawodowej",
                "Zaświadczenie o niekaralności",
                "Certyfikat ADR (przewóz materiałów niebezpiecznych)",
                "Uprawnienia HDS (hydrauliczny dźwig samochodowy)"
            ],
            "skills": [
                "Znajomość przepisów ruchu drogowego",
                "Obsługa tachografu cyfrowego",
                "Planowanie optymalnych tras",
                "Dbałość o stan techniczny pojazdu",
                "Zabezpieczanie ładunku",
                "Prowadzenie dokumentacji transportowej",
                "Obsługa GPS i systemów nawigacyjnych",
                "Podstawowa znajomość mechaniki pojazdowej"
            ],
            "achievements": [
                "Przejechanych X kilometrów bez wypadku",
                "Utrzymanie zużycia paliwa X% poniżej średniej firmowej",
                "Terminowość dostaw na poziomie X%",
                "Skrócenie czasu dostawy o X% dzięki optymalizacji trasy",
                "Bezbłędne prowadzenie dokumentacji przez X miesięcy",
                "Realizacja X dostaw miesięcznie",
                "Obsługa X stałych klientów z najwyższymi ocenami satysfakcji"
            ],
            "language_style": "Profesjonalny, konkretny, z naciskiem na bezpieczeństwo i odpowiedzialność"
        },
        "magazynier": {
            "certifications": [
                "Uprawnienia na wózki widłowe",
                "Uprawnienia na suwnice",
                "Certyfikat BHP",
                "Uprawnienia do obsługi systemów WMS"
            ],
            "skills": [
                "Obsługa skanerów i czytników kodów",
                "Kompletacja zamówień",
                "Inwentaryzacja",
                "Obsługa systemów magazynowych (WMS)",
                "Przyjmowanie i wydawanie towaru",
                "Kontrola jakościowa produktów",
                "Pakowanie i zabezpieczanie towaru"
            ],
            "achievements": [
                "Zwiększenie wydajności kompletacji o X%",
                "Zmniejszenie błędów w zamówieniach o X%",
                "Wdrożenie usprawnień w procesie X, co skróciło czas o Y%",
                "Bezbłędne przeprowadzenie X inwentaryzacji",
                "Utrzymanie 100% dokładności w zarządzaniu zapasami przez X miesięcy",
                "Obsługa X palet dziennie"
            ],
            "language_style": "Precyzyjny, operacyjny, podkreślający dokładność i efektywność"
        },
        "programista": {
            "certifications": [
                "Certyfikaty Microsoft/AWS/Google Cloud",
                "Certyfikaty językowe (Java, Python)",
                "Certyfikaty Agile/Scrum",
                "Certyfikat ITIL",
                "Certyfikat cyberbezpieczeństwa"
            ],
            "skills": [
                "Znajomość technologii X, Y, Z",
                "Tworzenie czystego, testowalnego kodu",
                "Projektowanie architektury systemów",
                "Testowanie i debugowanie aplikacji",
                "Praca z systemami kontroli wersji",
                "Współpraca w zespole programistycznym",
                "Code review",
                "Ciągła integracja/wdrażanie (CI/CD)"
            ],
            "achievements": [
                "Optymalizacja wydajności systemu X o Y%",
                "Skrócenie czasu ładowania aplikacji o X%",
                "Zredukowanie liczby błędów o X% poprzez wdrożenie testów automatycznych",
                "Wdrożenie X nowych funkcjonalności w ciągu Y miesięcy",
                "Przeprowadzenie refaktoryzacji kodu, co zmniejszyło jego złożoność o X%",
                "Stworzenie rozwiązania, które zaoszczędziło firmie X zł rocznie"
            ],
            "language_style": "Techniczny, analityczny, z wykorzystaniem specjalistycznej terminologii IT"
        },
        "sprzedawca": {
            "certifications": [
                "Certyfikat obsługi klienta",
                "Certyfikat sprzedażowy",
                "Uprawnienia do obsługi kasy fiskalnej",
                "Certyfikat z technik negocjacji"
            ],
            "skills": [
                "Profesjonalna obsługa klienta",
                "Znajomość technik sprzedaży",
                "Obsługa kasy fiskalnej i terminali płatniczych",
                "Zarządzanie zapasami na półkach",
                "Przygotowywanie ekspozycji produktów",
                "Rozwiązywanie problemów klientów",
                "Realizacja planów sprzedażowych"
            ],
            "achievements": [
                "Przekroczenie celu sprzedażowego o X%",
                "Zwiększenie średniej wartości koszyka o X%",
                "Pozyskanie X nowych stałych klientów",
                "Utrzymanie najwyższego wskaźnika satysfakcji klienta przez X miesięcy",
                "Przeprowadzenie X skutecznych akcji promocyjnych",
                "Wzrost sprzedaży w kategorii X o Y%"
            ],
            "language_style": "Nastawiony na klienta, entuzjastyczny, przekonujący"
        }
    }

    # Dodaj więcej ról w miarę potrzeb

    # Jeśli rola nie jest zdefiniowana, zwróć ogólne kompetencje
    if role not in role_competencies:
        return {
            "certifications": ["Certyfikaty branżowe", "Szkolenia specjalistyczne"],
            "skills": ["Umiejętności interpersonalne", "Organizacja pracy", "Rozwiązywanie problemów"],
            "achievements": ["Przekroczenie celów o X%", "Optymalizacja procesów", "Realizacja projektów"],
            "language_style": "Profesjonalny, rzeczowy, zorientowany na wyniki"
        }

    return role_competencies[role]

def get_job_type_template(job_type):
    """
    Get CV template guidance based on job type
    """
    templates = {
        "physical": """
    STRUKTURA CV DLA PRACY FIZYCZNEJ:
    1. Dane kontaktowe na górze strony
    2. Krótkie podsumowanie zawodowe (3-4 zdania)
    3. Uprawnienia i certyfikaty (na pierwszym miejscu)
    4. Doświadczenie zawodowe (konkretne dane liczbowe)
    5. Wykształcenie i kursy (zwięźle)
    6. Umiejętności techniczne i praktyczne (z podziałem na kategorie)

    FORMAT:
    - Maksymalnie 1-2 strony
    - Proste, czytelne formatowanie
    - Wypunktowania zamiast długich paragrafów
    - Podkreślenie uprawnień i kwalifikacji zawodowych

    STYL JĘZYKOWY:
    - Konkretne, rzeczowe sformułowania
    - Proste zdania bez żargonu
    - Nacisk na praktyczne umiejętności
    - Używanie czasowników czynnościowych (obsługiwałem, dostarczałem, naprawiałem)
    """,
        "technical": """
    STRUKTURA CV DLA PRACY TECHNICZNEJ:
    1. Dane kontaktowe i dane osobowe
    2. Podsumowanie zawodowe z kluczowymi umiejętnościami
    3. Kwalifikacje techniczne i uprawnienia
    4. Doświadczenie zawodowe z konkretnymi projektami
    5. Wykształcenie i specjalistyczne szkolenia
    6. Umiejętności techniczne z poziomem zaawansowania

    FORMAT:
    - 1-2 strony
    - Przejrzyste sekcje z podtytułami
    - Używanie tabel dla umiejętności technicznych
    - Uwypuklenie certyfikatów i uprawnień

    STYL JĘZYKOWY:
    - Precyzyjny, techniczny język
    - Szczegółowy opis umiejętności
    - Używanie branżowej terminologii
    - Konkretne osiągnięcia z liczbami i parametrami
    """,
        "office": """
    STRUKTURA CV DLA PRACY BIUROWEJ:
    1. Dane kontaktowe i profesjonalny profil
    2. Zwięzłe podsumowanie zawodowe
    3. Doświadczenie zawodowe (chronologicznie)
    4. Umiejętności biurowe i znajomość oprogramowania
    5. Wykształcenie i kursy
    6. Osiągnięcia i dodatkowe kwalifikacje

    FORMAT:
    - 1-2 strony
    - Eleganckie, czyste formatowanie
    - Spójne czcionki i marginesy
    - Umiarkowane używanie kolorów

    STYL JĘZYKOWY:
    - Profesjonalny, biznesowy język
    - Użycie czasowników biznesowych (koordynowałem, zarządzałem, analizowałem)
    - Podkreślenie umiejętności organizacyjnych i komunikacyjnych
    - Formalne, ale przystępne sformułowania
    """,
        "professional": """
    STRUKTURA CV DLA PRACY SPECJALISTYCZNEJ:
    1. Dane kontaktowe i profesjonalny profil
    2. Podsumowanie ekspertyz i kluczowych kompetencji
    3. Doświadczenie zawodowe z podkreśleniem osiągnięć
    4. Wykształcenie, specjalizacje i certyfikacje
    5. Publikacje, projekty badawcze lub specjalistyczne osiągnięcia
    6. Umiejętności specjalistyczne i znajomość metodologii

    FORMAT:
    - 2-3 strony
    - Profesjonalne, uporządkowane formatowanie
    - Możliwość dodania sekcji publikacji/projektów
    - Hierarchiczna organizacja informacji

    STYL JĘZYKOWY:
    - Zaawansowany, specjalistyczny język
    - Terminologia branżowa na wysokim poziomie
    - Podkreślenie ekspertyzy i autorytetu w dziedzinie
    - Uwypuklenie wartości dodanej dla organizacji
    """,
        "creative": """
    STRUKTURA CV DLA PRACY KREATYWNEJ:
    1. Dane kontaktowe i link do portfolio
    2. Kreatywne, wyróżniające się podsumowanie
    3. Wybrane projekty i osiągnięcia (przed doświadczeniem)
    4. Doświadczenie zawodowe
    5. Umiejętności kreatywne i techniczne
    6. Wykształcenie i rozwój kreatywny

    FORMAT:
    - 1-2 strony, ale z wyróżniającym się designem
    - Możliwość niestandardowego układu
    - Elementy graficzne podkreślające kreatywność
    - Więcej swobody w kolorach i formatowaniu

    STYL JĘZYKOWY:
    - Dynamiczny, kreatywny język
    - Balans między profesjonalizmem a kreatywnością
    - Uwypuklenie procesów kreatywnych i wyników
    - Unikalny, osobisty ton głosu
    """,
        "it": """
    STRUKTURA CV DLA PRACY W IT:
    1. Dane kontaktowe i linki (GitHub, LinkedIn)
    2. Zwięzłe podsumowanie techniczne
    3. Umiejętności techniczne pogrupowane według kategorii
    4. Doświadczenie zawodowe z konkretnymi projektami
    5. Wykształcenie i certyfikaty techniczne
    6. Projekty osobiste i open source

    FORMAT:
    - 1-2 strony
    - Techniczne, przejrzyste formatowanie
    - Tabele lub paski postępu dla umiejętności
    - Elementy kodu/pseudokodu jako akcenty

    STYL JĘZYKOWY:
    - Techniczny, precyzyjny język
    - Używanie terminologii IT
    - Konkretne metryki i rezultaty techniczne
    - Podkreślenie znajomości technologii i rozwiązanych problemów
    """
    }

    return templates.get(job_type, templates["office"])

def get_industry_specific_prompt(industry, seniority, job_type=None, specific_role=None):
    """
    Get industry-specific prompt guidance, enhanced with job type and role specifics
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
        "transport": """
    - Zastosuj precyzyjny język transportowy i logistyczny
    - Podkreśl znajomość przepisów transportowych i dokumentacji
    - Uwzględnij konkretne dane dotyczące realizowanych tras, ładunków, kilometrażu
    - Wyeksponuj znajomość procedur bezpieczeństwa i efektywności transportu
    - Podkreśl osiągnięcia w zakresie terminowości i jakości dostaw
    """,
        "retail": """
    - Użyj języka zorientowanego na klienta i sprzedaż
    - Podaj konkretne wyniki sprzedażowe i wskaźniki KPI
    - Wymień znajomość systemów kasowych i zarządzania zapasami
    - Podkreśl umiejętności w zakresie merchandisingu i układania ekspozycji
    - Uwzględnij osiągnięcia w zakresie obsługi klienta i rozwiązywania problemów
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

    # Dodaj wskazówki dotyczące typu pracy
    job_type_guidance = ""
    if job_type:
        job_type_template = get_job_type_template(job_type)
        job_type_guidance = f"\n\nWSKAZÓWKI DOTYCZĄCE TYPU PRACY ({job_type.upper()}):\n{job_type_template}"

    # Dodaj wskazówki dotyczące konkretnej roli
    role_guidance = ""
    if specific_role:
        competencies = get_role_specific_competencies(specific_role)

        role_guidance = f"\n\nWYMAGANE KOMPETENCJE DLA ROLI: {specific_role.upper()}\n"

        # Certyfikaty i uprawnienia
        role_guidance += "\nSugerowane certyfikaty i uprawnienia:\n"
        for cert in competencies.get("certifications", []):
            role_guidance += f"- {cert}\n"

        # Umiejętności
        role_guidance += "\nKluczowe umiejętności dla tej roli:\n"
        for skill in competencies.get("skills", []):
            role_guidance += f"- {skill}\n"

        # Typowe osiągnięcia
        role_guidance += "\nTypowe osiągnięcia w tej roli (zamień X, Y, Z na realne liczby):\n"
        for achievement in competencies.get("achievements", []):
            role_guidance += f"- {achievement}\n"

        # Styl języka
        role_guidance += f"\nSugerowany styl języka: {competencies.get('language_style', 'Profesjonalny')}\n"

    result = industry_guidance + "\n" + seniority_guidance.get(seniority, seniority_guidance["mid"])

    # Dodaj dodatkowe wskazówki, jeśli są dostępne
    if job_type_guidance:
        result += job_type_guidance
    if role_guidance:
        result += role_guidance

    return result

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
    - Każde doświadczenie zawodowe opisz w 4-6 punktjson" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("