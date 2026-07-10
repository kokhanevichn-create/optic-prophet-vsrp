(() => {
  const input = document.getElementById("photo-input");
  const grid = document.getElementById("preview-grid");
  const dropzone = document.getElementById("dropzone");
  if (!input || !grid || !dropzone) return;

  const render = (files) => {
    grid.innerHTML = "";
    if (!files || !files.length) {
      grid.hidden = true;
      return;
    }
    grid.hidden = false;
    [...files].slice(0, 8).forEach((file, i) => {
      const img = document.createElement("img");
      img.alt = `Preview ${i + 1}`;
      img.style.animationDelay = `${i * 60}ms`;
      img.src = URL.createObjectURL(file);
      grid.appendChild(img);
    });
  };

  input.addEventListener("change", () => render(input.files));

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });
  });
  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
    });
  });
  dropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer?.files;
    if (!files?.length) return;
    input.files = files;
    render(files);
  });
})();
