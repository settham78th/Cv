/**
 * Main JavaScript file for the PDF Text Processor application
 */

document.addEventListener('DOMContentLoaded', function() {
    // Track page load time for performance monitoring
    const pageLoadTime = window.performance.timing.domContentLoadedEventEnd - 
                        window.performance.timing.navigationStart;
    console.log(`Page loaded in ${pageLoadTime}ms`);
    
    // File input validation
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            validateFileInput(this);
        });
    });
    
    // Form submission handling
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const fileInputs = this.querySelectorAll('input[type="file"]');
            
            fileInputs.forEach(input => {
                if (!validateFileInput(input)) {
                    e.preventDefault();
                }
            });
        });
    });
    
    // Activate all tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Add alert auto-dismiss
    const alerts = document.querySelectorAll('.alert:not(.alert-permanent)');
    
    alerts.forEach(alert => {
        setTimeout(() => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
});

/**
 * Validate file input to ensure it's a PDF
 * @param {HTMLInputElement} fileInput - The file input element to validate
 * @returns {boolean} - Whether the file is valid
 */
function validateFileInput(fileInput) {
    if (fileInput.files.length === 0) {
        return false;
    }
    
    const file = fileInput.files[0];
    const filename = file.name.toLowerCase();
    
    // Check file extension
    if (!filename.endsWith('.pdf')) {
        alert('Please select a PDF file');
        fileInput.value = '';
        return false;
    }
    
    // Check file size (max 20MB)
    const maxSize = 20 * 1024 * 1024; // 20MB in bytes
    if (file.size > maxSize) {
        alert('File size exceeds the limit (20MB)');
        fileInput.value = '';
        return false;
    }
    
    return true;
}

/**
 * Toggle visibility of an element
 * @param {string} elementId - The ID of the element to toggle
 */
function toggleElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        element.classList.toggle('d-none');
    }
}

/**
 * Show a notification to the user
 * @param {string} message - The message to display
 * @param {string} type - The type of notification (success, danger, warning, info)
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show`;
    notification.setAttribute('role', 'alert');
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to the DOM
    const container = document.querySelector('main.container');
    container.insertBefore(notification, container.firstChild);
    
    // Auto dismiss after 5 seconds
    setTimeout(() => {
        const bsAlert = new bootstrap.Alert(notification);
        bsAlert.close();
    }, 5000);
}
