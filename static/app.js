document.getElementById('input-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const text = document.getElementById('text').value;
    const files = document.getElementById('files').files;

    const formData = new FormData();
    formData.append('text', text);

    for (let i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    const response = await fetch('http://127.0.0.1:8000/process', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    console.log(result);
    document.getElementById('result').innerText = result.message;
});

document.getElementById('text').addEventListener('input', function(event) {
    toggleSubmitButton(event);
});
document.getElementById('files').addEventListener('change', function(event) {
    toggleSubmitButton(event);
});

function toggleSubmitButton(event) {
    const textInput = document.getElementById('text');
    const fileInput = document.getElementById('files');
    const submitButton = document.getElementById('submit-button');
    const fileNameElement = document.getElementById("fileName");

    const hasText = textInput.value.trim() !== '';
    const hasFile = fileInput.files.length > 0;
    submitButton.disabled = !(hasText || hasFile);
    textInput.disabled = hasFile;
    fileInput.disabled = hasText;

    if (event.target.id === 'files') {
        if (hasFile) {
            const fileName = fileInput.files[0].name;
            fileNameElement.textContent = `Selected file: ${fileName}`;
        } else {
            fileNameElement.textContent = "";
        }
    }

    if (event.target.id === 'text') {
        if (!hasText) {
            fileNameElement.textContent = "";
        }
    }
}


// document
//   .getElementById("files")
//   .addEventListener("change", function (event) {
//     const fileInput = event.target;
//     const fileNameElement = document.getElementById("fileName");
//     const text = document.getElementById('text');
//     if (fileInput.files.length > 0) {
//       const fileName = fileInput.files[0].name;
//       fileNameElement.textContent = `Selected file: ${fileName}`;
//       text.disabled = true
//     } 
//     else {
//       fileNameElement.textContent = "";
//       text.disabled = false;
//     }
//   });
