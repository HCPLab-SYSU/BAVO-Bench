document.addEventListener('DOMContentLoaded', () => {
  const button = document.getElementById('copy-bibtex');
  const code = document.querySelector('#bibtex-code code');
  if (!button || !code) return;

  button.addEventListener('click', async () => {
    const citation = code.textContent.trim();
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(citation);
      } else {
        const field = document.createElement('textarea');
        field.value = citation;
        field.style.position = 'fixed';
        field.style.opacity = '0';
        document.body.appendChild(field);
        field.select();
        const copied = document.execCommand('copy');
        field.remove();
        if (!copied) throw new Error('Copy failed');
      }
      button.textContent = 'Copied!';
      window.setTimeout(() => { button.textContent = 'Copy BibTeX'; }, 2000);
    } catch {
      button.textContent = 'Select text to copy';
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(code);
      selection.removeAllRanges();
      selection.addRange(range);
    }
  });
});
