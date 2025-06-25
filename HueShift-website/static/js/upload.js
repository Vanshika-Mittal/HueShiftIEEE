async function handleImageUpload(event) {
    event.preventDefault();
    const formData = new FormData(event.target);
    const fileInput = document.querySelector('input[type="file"]');
    const file = fileInput.files[0];

    if (!file) {
        showAlert('Please select a file first.', 'danger');
        return;
    }

    try {
        showAlert('Uploading and processing your image...', 'info');
        const response = await fetch('/colorize/image', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const data = await response.json();
        if (data.status === 'success') {
            showAlert('Image processed successfully! Redirecting to gallery...', 'success');
            // Store the processed image URLs
            sessionStorage.setItem('lastProcessedImage', JSON.stringify({
                resultUrl: data.result_url,
                thumbnailUrl: data.thumbnail_url,
                imageId: data.image_id
            }));
            window.location.href = '/gallery';
        }
    } catch (error) {
        showAlert('Error processing image: ' + error.message, 'danger');
    }
}

async function checkProcessingStatus(filename) {
    try {
        const response = await fetch(`/check-status/${filename}`);
        const data = await response.json();
        
        if (data.status === 'completed') {
            const processedUrl = sessionStorage.getItem('processedImageUrl');
            window.location.href = `/gallery?processed=${encodeURIComponent(processedUrl)}`;
        } else if (data.status === 'processing') {
            // Check again in 5 seconds
            setTimeout(() => checkProcessingStatus(filename), 5000);
        } else {
            showAlert('Processing failed: ' + data.error, 'danger');
        }
    } catch (error) {
        showAlert('Error checking status: ' + error.message, 'danger');
    }
} 