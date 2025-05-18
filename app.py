import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import tempfile
import uuid

from pdf_extractor import extract_text_from_pdf
from ai_processor import process_with_openrouter
from config import configure_logging

# Create and configure the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", str(uuid.uuid4()))

# Configure logging
configure_logging(app)
logger = logging.getLogger(__name__)

# Configure upload settings
ALLOWED_EXTENSIONS = {'pdf'}
TEMP_FOLDER = tempfile.gettempdir()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_pdf():
    # Check if the post request has the file part
    if 'pdf_file' not in request.files:
        flash('No file part', 'danger')
        logger.warning("File upload attempted without file part")
        return redirect(url_for('index'))
    
    file = request.files['pdf_file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file', 'danger')
        logger.warning("File upload attempted with empty filename")
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        # Save to temporary file
        filename = secure_filename(file.filename)
        temp_path = os.path.join(TEMP_FOLDER, filename)
        file.save(temp_path)
        
        logger.info(f"Processing PDF file: {filename}")
        
        try:
            # Extract text from PDF
            extracted_text = extract_text_from_pdf(temp_path)
            
            if not extracted_text or extracted_text.strip() == "":
                flash('Could not extract text from PDF. It might be empty or scanned.', 'warning')
                logger.warning(f"Empty text extracted from {filename}")
                return redirect(url_for('index'))
            
            # Process with OpenRouter
            prompt = request.form.get('prompt', 'Summarize the following text')
            ai_prompt = f"{prompt}:\n\n{extracted_text[:2000]}"  # Limit text length for API
            
            ai_result = process_with_openrouter(ai_prompt)
            
            # Store in session for display
            session['extracted_text'] = extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else "")
            session['ai_result'] = ai_result
            session['filename'] = filename
            
            logger.info(f"Successfully processed {filename}")
            
            # Clean up
            os.remove(temp_path)
            
            return redirect(url_for('result'))
            
        except Exception as e:
            flash(f'Error processing PDF: {str(e)}', 'danger')
            logger.error(f"Error processing {filename}: {str(e)}", exc_info=True)
            
            # Clean up on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            return redirect(url_for('index'))
    else:
        flash('File type not allowed. Please upload a PDF file.', 'danger')
        logger.warning(f"Invalid file type attempted: {file.filename}")
        return redirect(url_for('index'))

@app.route('/result')
def result():
    # Check if we have results in the session
    if 'extracted_text' not in session or 'ai_result' not in session:
        flash('No results to display. Please process a PDF first.', 'warning')
        logger.warning("Access to results page without processing data")
        return redirect(url_for('index'))
    
    return render_template(
        'result.html',
        filename=session.get('filename', 'Unknown file'),
        extracted_text=session.get('extracted_text', ''),
        ai_result=session.get('ai_result', '')
    )

@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 error: {request.url}")
    return render_template('index.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500 error: {str(e)}", exc_info=True)
    flash('An unexpected error occurred. Please try again later.', 'danger')
    return render_template('index.html'), 500
