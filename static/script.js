document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('imageUpload');
    const startCameraButton = document.getElementById('startCameraButton');
    const imageCanvas = document.getElementById('imageCanvas');
    const webcamVideo = document.getElementById('webcamVideo');
    const processButton = document.getElementById('processButton');
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    const ctx = imageCanvas.getContext('2d');

    let currentImage = null; // Stores the image data (Image object or canvas data)

    // Function to draw an image onto the canvas, fitting it and maintaining aspect ratio
    function drawImageOnCanvas(img) {
        const maxWidth = 700; // Max width for the canvas display
        const maxHeight = 400; // Max height for the canvas display

        let width = img.width;
        let height = img.height;

        // Calculate new dimensions to fit within max bounds
        if (width > maxWidth) {
            height = height * (maxWidth / width);
            width = maxWidth;
        }
        if (height > maxHeight) {
            width = width * (maxHeight / height);
            height = maxHeight;
        }

        imageCanvas.width = width;
        imageCanvas.height = height;
        ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
        ctx.drawImage(img, 0, 0, width, height);
        currentImage = img; // Store the original image
        processButton.disabled = false;
    }

    // --- Image Upload ---
    imageUpload.addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    webcamVideo.style.display = 'none'; // Hide video if showing
                    drawImageOnCanvas(img);
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
    });

    // --- Webcam Capture ---
    startCameraButton.addEventListener('click', async () => {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                webcamVideo.srcObject = stream;
                webcamVideo.play();
                webcamVideo.style.display = 'block';
                imageCanvas.style.display = 'none'; // Hide canvas while video is playing
                processButton.disabled = true; // Disable process until photo is taken
                startCameraButton.textContent = 'Take Photo'; // Change button text

                // This re-assigns the click listener for the 'Take Photo' action
                startCameraButton.onclick = () => {
                    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
                    imageCanvas.width = webcamVideo.videoWidth;
                    imageCanvas.height = webcamVideo.videoHeight;
                    ctx.drawImage(webcamVideo, 0, 0, imageCanvas.width, imageCanvas.height);
                    imageCanvas.style.display = 'block';
                    webcamVideo.style.display = 'none';
                    stream.getTracks().forEach(track => track.stop()); // Stop webcam stream
                    currentImage = imageCanvas; // Store the canvas content
                    processButton.disabled = false;
                    startCameraButton.textContent = 'Take Photo'; // Reset button text
                    // Re-add the original event listener behavior (for starting camera again)
                    startCameraButton.onclick = null; // Clear this temporary handler
                    startCameraButton.addEventListener('click', async () => { /* Original logic */ }); // (simplified, actual re-binding is more robust)
                };

            } catch (err) {
                console.error("Error accessing webcam: ", err);
                alert("Could not access webcam. Please ensure you have a camera and grant permission.");
            }
        } else {
            alert("Webcam not supported by your browser.");
        }
    });

    // --- Process Button ---
    processButton.addEventListener('click', async () => {
        if (!currentImage) {
            alert("Please upload an image or take a photo first.");
            return;
        }

        loadingDiv.style.display = 'block';
        resultsDiv.innerHTML = '<p>Processing...</p>';
        processButton.disabled = true;

        // Convert canvas content to Blob (image file)
        imageCanvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'handwriting.png'); // Send as a PNG file

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData,
                });

                const data = await response.json();

                if (response.ok) {
                    displayResults(data.results);
                } else {
                    resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.error || 'Unknown error'}</p>`;
                }
            } catch (error) {
                console.error('Error during prediction:', error);
                resultsDiv.innerHTML = `<p style="color: red;">Network or server error: ${error.message}</p>`;
            } finally {
                loadingDiv.style.display = 'none';
                processButton.disabled = false;
            }
        }, 'image/png'); // Specify PNG format
    });

    function displayResults(results) {
        resultsDiv.innerHTML = ''; // Clear previous results
        if (results && results.length > 0) {
            results.forEach(item => {
                const p = document.createElement('p');
                p.classList.add('result-item');
                p.innerHTML = `<strong>Detected:</strong> <span>${item.predicted_text}</span> <br> <strong>Closest Match:</strong> <span>${item.closest_match}</span>`;
                resultsDiv.appendChild(p);
            });
        } else {
            resultsDiv.innerHTML = '<p>No words detected or no results.</p>';
        }
    }
});