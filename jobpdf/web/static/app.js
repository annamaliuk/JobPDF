// JobPDF upload page (JM-43). Optional: the form works without this script.
// It checks the file before sending it and shows that the CV is being read.
(function () {
  "use strict";
  var form = document.getElementById("upload");
  if (!form) return;
  var input = form.querySelector('input[type="file"]');
  var button = form.querySelector("button");
  var status = document.getElementById("status");
  var maxBytes = Number(form.getAttribute("data-max-bytes")) || 0;
  var allowed = /\.(pdf|docx)$/i;

  function say(text, problem) {
    status.textContent = text;
    status.className = problem ? "status problem" : "status";
  }

  function problem() {
    var file = input.files && input.files[0];
    if (!file) return "Choose a PDF or DOCX file.";
    if (!allowed.test(file.name)) return "Only PDF and DOCX files are accepted.";
    if (maxBytes && file.size > maxBytes) {
      return "The file is larger than " + Math.floor(maxBytes / 1048576) + " MB.";
    }
    return "";
  }

  input.addEventListener("change", function () {
    var found = input.files && input.files.length ? problem() : "";
    say(found, Boolean(found));
  });

  form.addEventListener("submit", function (event) {
    var found = problem();
    if (found) {
      event.preventDefault();
      say(found, true);
      return;
    }
    button.disabled = true;
    say("Reading your CV. This usually takes 10 to 30 seconds.", false);
  });

  // Coming back with the browser's Back button restores the page as it was left.
  window.addEventListener("pageshow", function () {
    button.disabled = false;
    say("", false);
  });
})();
