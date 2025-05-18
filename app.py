import os
import logging
from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import tempfile
from utils.pdf_extraction import extract_text_from_pdf
from utils.openrouter_api import (
    optimize_cv, 
    optimize_cv_with_keywords,
    generate_recruiter_feedback,
    generate_cover_letter,
    translate_to_english,
    suggest_alternative_careers,
    generate_multi_versions,
    analyze_job_url,
    analyze_market_trends,
    ats_optimization_check,
    generate_interview_questions,
    extract_keywords_from_job,
    generate_keywords_html
)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Create a temporary directory for storing uploaded files
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-cv', methods=['POST'])
def upload_cv():
    # Check if the post request has the file part
    if 'cv_file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['cv_file']
    
    # If user does not select file, browser submits empty file without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        try:
            # Extract text from PDF
            cv_text = extract_text_from_pdf(file_path)
            
            # Store the CV text in session
            session['cv_text'] = cv_text
            session['original_filename'] = filename
            
            # Remove the file after extraction
            os.remove(file_path)
            
            return jsonify({
                'success': True, 
                'cv_text': cv_text,
                'message': 'CV successfully uploaded and text extracted.'
            })
        
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            # Clean up the file in case of error
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({
                'success': False,
                'message': f"Error processing PDF: {str(e)}"
            }), 500
    
    return jsonify({
        'success': False,
        'message': 'Invalid file type. Please upload a PDF file.'
    }), 400

@app.route('/process-cv', methods=['POST'])
def process_cv():
    data = request.json
    cv_text = data.get('cv_text') or session.get('cv_text')
    job_description = data.get('job_description', '')
    job_url = data.get('job_url', '')
    selected_option = data.get('selected_option', '')
    roles = data.get('roles', [])
    
    if not cv_text:
        return jsonify({
            'success': False,
            'message': 'No CV text found. Please upload a CV first.'
        }), 400
    
    # Process job URL if provided
    extracted_job_description = ''
    if job_url and not job_description:
        try:
            extracted_job_description = analyze_job_url(job_url)
            job_description = extracted_job_description
        except Exception as e:
            logger.error(f"Error extracting job description from URL: {str(e)}")
            return jsonify({
                'success': False,
                'message': f"Error extracting job description from URL: {str(e)}"
            }), 500
    
    # Process according to selected option
    try:
        result = None
        job_title = data.get('job_title', '')
        industry = data.get('industry', '')
        
        if selected_option == 'optimize':
            result = optimize_cv(cv_text, job_description)
        elif selected_option == 'feedback':
            result = generate_recruiter_feedback(cv_text, job_description)
        elif selected_option == 'cover_letter':
            result = generate_cover_letter(cv_text, job_description)
        elif selected_option == 'translate':
            result = translate_to_english(cv_text)
        elif selected_option == 'alternative_careers':
            result = suggest_alternative_careers(cv_text)
        elif selected_option == 'multi_versions':
            result = generate_multi_versions(cv_text, roles)
        elif selected_option == 'ats_check':
            result = ats_optimization_check(cv_text, job_description)
        elif selected_option == 'interview_questions':
            result = generate_interview_questions(cv_text, job_description)
        elif selected_option == 'market_trends':
            result = analyze_market_trends(job_title, industry)
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid option selected.'
            }), 400
        
        return jsonify({
            'success': True,
            'result': result,
            'job_description': extracted_job_description if extracted_job_description else None
        })
    
    except Exception as e:
        logger.error(f"Error processing CV: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"Error processing request: {str(e)}"
        }), 500

@app.route('/keywords', methods=['GET', 'POST'])
def keywords():
    if request.method == 'POST':
        job_description = request.form.get('job_description', '').strip()
        
        if not job_description:
            flash('Proszę podać opis stanowiska', 'danger')
            return render_template('keywords.html')
        
        try:
            # Ekstrakcja słów kluczowych z opisu stanowiska
            keywords_data = extract_keywords_from_job(job_description)
            keywords_html = generate_keywords_html(keywords_data)
            
            # Zapisz opis stanowiska w sesji do późniejszego wykorzystania
            session['job_description'] = job_description
            session['keywords_data'] = keywords_data
            
            return render_template('keywords_result.html', 
                                  keywords_html=keywords_html,
                                  job_description=job_description)
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}", exc_info=True)
            flash(f'Wystąpił błąd podczas analizy opisu stanowiska: {str(e)}', 'danger')
            return render_template('keywords.html')
    
    return render_template('keywords.html')

@app.route('/optimize-with-keywords', methods=['POST'])
def optimize_with_keywords():
    cv_text = request.form.get('cv_text', '').strip()
    job_description = session.get('job_description', '')
    keywords_data = session.get('keywords_data', {})
    
    if not cv_text:
        flash('Proszę podać treść CV', 'danger')
        return redirect(url_for('keywords'))
    
    if not job_description:
        flash('Brak opisu stanowiska. Proszę najpierw przeanalizować opis stanowiska.', 'danger')
        return redirect(url_for('keywords'))
    
    try:
        # Optymalizuj CV z uwzględnieniem słów kluczowych
        optimized_cv = optimize_cv_with_keywords(cv_text, job_description, keywords_data)
        
        return render_template('optimized_cv.html',
                              original_cv=cv_text,
                              optimized_cv=optimized_cv,
                              job_description=job_description,
                              keywords_html=generate_keywords_html(keywords_data))
    
    except Exception as e:
        logger.error(f"Error optimizing CV with keywords: {str(e)}", exc_info=True)
        flash(f'Wystąpił błąd podczas optymalizacji CV: {str(e)}', 'danger')
        return redirect(url_for('keywords'))

@app.route('/keywords-json', methods=['POST'])
def keywords_json():
    data = request.json
    job_description = data.get('job_description', '')
    
    if not job_description:
        return jsonify({
            'success': False,
            'message': 'Brak opisu stanowiska'
        }), 400
    
    try:
        # Ekstrakcja słów kluczowych z opisu stanowiska
        keywords_data = extract_keywords_from_job(job_description)
        keywords_html = generate_keywords_html(keywords_data)
        
        return jsonify({
            'success': True,
            'keywords_data': keywords_data,
            'keywords_html': keywords_html
        })
    
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'message': f"Błąd podczas analizy opisu stanowiska: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
