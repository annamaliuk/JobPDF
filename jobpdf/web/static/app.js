// JobPDF upload page (JM-43, progress steps JM-44). Optional: the form works without this script.
// It checks the file before sending it. Where the browser can stream, it posts the CV itself and
// shows the steps of the analysis as they happen, then puts the report on this page. Nothing is
// kept in the browser: no storage, no cookie, and the only request goes to the form's own address.
(function () {
  "use strict";
  var form = document.getElementById("upload");
  if (!form) return;
  var input = form.querySelector('input[type="file"]');
  var button = form.querySelector("button");
  var status = document.getElementById("status");
  var panel = document.getElementById("progress");
  var maxBytes = Number(form.getAttribute("data-max-bytes")) || 0;
  var allowed = /\.(pdf|docx)$/i;
  var SLOW_AFTER_S = 45;
  var STATE_WORDS = { waiting: "Waiting", active: "In progress", done: "Done" };
  var canStream = Boolean(
    window.fetch && window.ReadableStream && window.TextDecoder && window.FormData &&
    window.DOMParser && panel
  );
  var timer = null;

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

  // --- the steps ---------------------------------------------------------------------------

  function steps() {
    return Array.prototype.slice.call(panel.querySelectorAll(".step"));
  }

  function setState(item, state) {
    item.setAttribute("data-state", state);
    item.querySelector(".state").textContent = STATE_WORDS[state];
  }

  function showStep(id) {
    var list = steps();
    var found = false;
    list.forEach(function (item, index) {
      if (item.getAttribute("data-step") === id) {
        found = true;
        setState(item, "active");
        document.getElementById("progress-note").textContent =
          "Step " + (index + 1) + " of " + list.length + ": " + item.querySelector(".label").textContent;
      } else if (!found) {
        setState(item, "done");
      } else {
        setState(item, "waiting");
      }
    });
  }

  function setIdleHidden(hide) {
    Array.prototype.forEach.call(document.querySelectorAll(".idle"), function (node) {
      node.hidden = hide;
    });
  }

  function startProgress() {
    steps().forEach(function (item) { setState(item, "waiting"); });
    document.getElementById("slow").hidden = true;
    document.getElementById("progress-retry").hidden = true;
    document.getElementById("progress-heading").textContent = "Reading your CV";
    document.getElementById("elapsed").textContent = "0";
    setIdleHidden(true);
    panel.hidden = false;
    document.getElementById("progress-heading").focus();
    var started = Date.now();
    timer = window.setInterval(function () {
      var seconds = Math.floor((Date.now() - started) / 1000);
      document.getElementById("elapsed").textContent = String(seconds);
      document.getElementById("slow").hidden = seconds < SLOW_AFTER_S;
    }, 1000);
  }

  function stopTimer() {
    if (timer !== null) window.clearInterval(timer);
    timer = null;
  }

  function stopWithMessage(message) {
    stopTimer();
    steps().forEach(function (item) {
      if (item.getAttribute("data-state") === "active") setState(item, "waiting");
    });
    document.getElementById("progress-heading").textContent = "That did not work";
    document.getElementById("slow").hidden = true;
    document.getElementById("progress-note").textContent = message;
    var slow = document.getElementById("slow");
    slow.textContent = message;
    slow.hidden = false;
    document.getElementById("progress-retry").hidden = false;
  }

  // Replace this page's content with a finished page sent by the server (the report or an
  // error page). It is parsed as data: nothing in it runs.
  function showPage(html) {
    stopTimer();
    var next = new DOMParser().parseFromString(html, "text/html");
    var main = document.getElementById("main");
    var incoming = next.getElementById("main");
    if (!main || !incoming) {
      stopWithMessage("The page could not be shown. Try again.");
      return;
    }
    document.title = next.title;
    var adopted = document.adoptNode(incoming);
    main.replaceWith(adopted);
    adopted.setAttribute("tabindex", "-1");
    adopted.focus();
    window.scrollTo(0, 0);
  }

  // --- the stream --------------------------------------------------------------------------

  function handleBlock(block) {
    var event = "message";
    var data = [];
    block.split("\n").forEach(function (line) {
      if (line.indexOf("event:") === 0) event = line.slice(6).trim();
      else if (line.indexOf("data:") === 0) data.push(line.slice(5).replace(/^ /, ""));
    });
    if (!data.length) return null; // a comment such as the keep-alive line
    var payload;
    try { payload = JSON.parse(data.join("\n")); } catch (e) { return null; }
    if (event === "step") {
      showStep(payload.step);
      return null;
    }
    if (event === "done" || event === "failed") return payload;
    return null;
  }

  function readStream(response) {
    var reader = response.body.getReader();
    var decoder = new TextDecoder("utf-8");
    var buffer = "";
    var finished = false;

    function finish(payload) {
      finished = true;
      if (payload && typeof payload.html === "string") showPage(payload.html);
      else stopWithMessage("The analysis failed. Try again.");
    }

    function pump() {
      return reader.read().then(function (chunk) {
        if (chunk.done) {
          if (!finished) stopWithMessage("The connection was lost before the report was ready. Try again.");
          return;
        }
        buffer += decoder.decode(chunk.value, { stream: true }).replace(/\r\n/g, "\n");
        var cut;
        while (!finished && (cut = buffer.indexOf("\n\n")) !== -1) {
          var payload = handleBlock(buffer.slice(0, cut));
          buffer = buffer.slice(cut + 2);
          if (payload) finish(payload);
        }
        if (finished) {
          reader.cancel();
          return;
        }
        return pump();
      });
    }
    return pump();
  }

  function submitStreaming() {
    startProgress();
    fetch(form.action, {
      method: "POST",
      body: new FormData(form),
      headers: { Accept: "text/event-stream" },
      credentials: "omit",
      cache: "no-store"
    }).then(function (response) {
      var type = response.headers.get("content-type") || "";
      if (type.indexOf("text/event-stream") === 0 && response.body) return readStream(response);
      // Refused before the analysis began (too many uploads, busy, bad file): a finished page.
      return response.text().then(showPage);
    }).catch(function () {
      stopWithMessage("Could not reach the server. Check your connection and try again.");
    });
  }

  // --- the form ----------------------------------------------------------------------------

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
    if (canStream) {
      event.preventDefault();
      submitStreaming();
      return;
    }
    say("Reading your CV. This usually takes 10 to 30 seconds.", false);
  });

  // Coming back with the browser's Back button restores the page as it was left.
  window.addEventListener("pageshow", function () {
    stopTimer();
    button.disabled = false;
    say("", false);
    if (panel) {
      panel.hidden = true;
      setIdleHidden(false);
    }
  });
})();
