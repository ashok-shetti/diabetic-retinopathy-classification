document.addEventListener('DOMContentLoaded', () => {
  const dropzone = document.getElementById('dropzone');
  const imageInput = document.getElementById('imageInput');
  const browseBtn = document.getElementById('browseBtn');
  const previewContainer = document.getElementById('previewContainer');
  const previewImage = document.getElementById('previewImage');
  const dropzoneContent = document.getElementById('dropzoneContent');
  const fileInfo = document.getElementById('fileInfo');
  const submitBtn = document.getElementById('submitBtn');
  const predictionForm = document.getElementById('predictionForm');
  const loadingOverlay = document.getElementById('loadingOverlay');

  if (dropzone && imageInput) {
    // Click on dropzone area triggers file picker
    dropzone.addEventListener('click', (e) => {
      if (e.target !== browseBtn && !browseBtn.contains(e.target)) {
        imageInput.click();
      }
    });

    if (browseBtn) {
      browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        imageInput.click();
      });
    }

    // Drag and Drop Event Listeners
    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      }, false);
    });

    // Handle dropped files
    dropzone.addEventListener('drop', (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        imageInput.files = files;
        handleFile(files[0]);
      }
    }, false);

    // Handle file selected via browser dialogue
    imageInput.addEventListener('change', () => {
      if (imageInput.files.length > 0) {
        handleFile(imageInput.files[0]);
      }
    });

    // Render image preview and display file information
    function handleFile(file) {
      if (!file || !file.type.startsWith('image/')) {
        alert('Please select a valid image file (PNG, JPG, JPEG).');
        return;
      }

      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onloadend = () => {
        previewImage.src = reader.result;
        previewImage.style.display = 'block';
        previewContainer.classList.add('active');
        dropzoneContent.style.display = 'none';
        
        const fileSize = (file.size / 1024).toFixed(1);
        fileInfo.textContent = `${file.name} (${fileSize} KB)`;
        
        if (submitBtn) {
          submitBtn.removeAttribute('disabled');
        }
      };
    }
  }

  // Handle Form Submission and Display Loading State Overlay
  if (predictionForm && loadingOverlay) {
    predictionForm.addEventListener('submit', () => {
      loadingOverlay.classList.add('active');
    });
  }
});
