document.addEventListener("DOMContentLoaded", function() {
    const viewer = document.getElementById('pdf-viewer');
    const zoomInButton = document.getElementById('zoom-in');
    const zoomOutButton = document.getElementById('zoom-out');

    // Load the converted PDF content
    fetch('sat.html') 
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.text();
        })
        .then(data => {
            viewer.innerHTML = data;
        })
        .catch(error => {
            console.error('Error loading PDF content:', error);
            viewer.innerHTML = `<p>Failed to load PDF content: ${error.message}</p>`;
        });

    // Zoom functions
    zoomInButton.addEventListener('click', function() {
        adjustFontSize(1.2);
    });

    zoomOutButton.addEventListener('click', function() {
        adjustFontSize(0.8);
    });

    function adjustFontSize(scale) {
        const elements = viewer.querySelectorAll('*');
        elements.forEach(el => {
            const computedStyle = window.getComputedStyle(el, null);
            const fontSize = computedStyle.getPropertyValue('font-size');
            const newSize = parseFloat(fontSize) * scale;
            el.style.fontSize = newSize + 'px';
        });
    }
});