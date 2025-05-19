document.addEventListener('DOMContentLoaded', function() {
    // File input validation
    const fileInput = document.getElementById('resume');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Validate file type
                if (!file.type.match('application/pdf')) {
                    alert('Only PDF files are allowed');
                    e.target.value = '';
                    return;
                }
                
                // Validate file size (5MB max)
                if (file.size > 5 * 1024 * 1024) {
                    alert('File size exceeds 5MB limit');
                    e.target.value = '';
                    return;
                }
            }
        });
    }
    
    // Add animation to result cards
    const resultCards = document.querySelectorAll('.card');
    if (resultCards.length > 0) {
        resultCards.forEach((card, index) => {
            setTimeout(() => {
                card.classList.add('fade-in');
            }, index * 100);
        });
    }
    
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    if (tooltipTriggerList.length > 0) {
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
});
