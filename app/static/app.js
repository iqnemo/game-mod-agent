const state = {
  filters: [],
  debugMode: false,
};

const gameSelect = document.getElementById("game-select");
const modsList = document.getElementById("mods-list");
const includeBaseGame = document.getElementById("include-base-game");
const questionInput = document.getElementById("question");
const sourceTypeSelect = document.getElementById("source-type");
const sendButton = document.getElementById("send-button");
const debugButton = document.getElementById("debug-button");
const answerEl = document.getElementById("answer");
const sourcesEl = document.getElementById("sources");
const statusEl = document.getElementById("status");

function selectedMods() {
  return Array.from(modsList.querySelectorAll('input[type="checkbox"]:checked')).map((input) => input.value);
}

function selectedGameEntry() {
  const game = gameSelect.value;
  return state.filters.find((entry) => entry.game === game) || null;
}

function renderMods() {
  const entry = selectedGameEntry();
  modsList.innerHTML = "";

  if (!entry || entry.mods.length === 0) {
    modsList.innerHTML = '<p class="empty">No mod options for this game yet.</p>';
    return;
  }

  entry.mods.forEach((mod) => {
    const id = `mod-${mod.toLowerCase().replace(/[^a-z0-9]+/g, "-")}`;
    const label = document.createElement("label");
    label.className = "mod-chip";
    label.innerHTML = `<input type="checkbox" id="${id}" value="${mod}"><span>${mod}</span>`;
    modsList.appendChild(label);
  });
}

function renderGames() {
  state.filters.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.game;
    option.textContent = entry.game;
    gameSelect.appendChild(option);
  });
}

function setStatus(message) {
  statusEl.textContent = message;
}

function renderAnswer(text) {
  answerEl.textContent = text;
  answerEl.classList.toggle("empty", !text);
}

function renderSources(sources) {
  if (!sources || sources.length === 0) {
    sourcesEl.textContent = "No sources returned.";
    sourcesEl.classList.add("empty");
    return;
  }

  sourcesEl.classList.remove("empty");
  sourcesEl.innerHTML = sources
    .map((source) => {
      const scope = [source.game, ...(source.mods || [])].filter(Boolean).join(" / ");
      const link = source.canonical_uri
        ? `<p class="source-meta"><a class="source-link" href="${source.canonical_uri}" target="_blank" rel="noreferrer">${source.canonical_uri}</a></p>`
        : "";
      const title = source.title || `${source.source_type || "source"} [${source.index}]`;
      return `
        <article class="source-card">
          <p class="source-title">[${source.index}] ${title}</p>
          <p class="source-meta">${scope || "Unscoped"}${source.author_name ? ` • ${source.author_name}` : ""}</p>
          ${link}
          <p class="source-snippet">${source.snippet || ""}</p>
        </article>
      `;
    })
    .join("");
}

async function loadFilters() {
  const response = await fetch("/api/filters");
  if (!response.ok) {
    throw new Error("Failed to load filters");
  }

  const payload = await response.json();
  state.filters = payload.games || [];
  renderGames();
  renderMods();
}

async function submitQuestion() {
  const question = questionInput.value.trim();
  if (!question) {
    setStatus("Enter a question first.");
    return;
  }

  const payload = {
    question,
    game: gameSelect.value || null,
    mods: selectedMods(),
    include_base_game: includeBaseGame.checked,
    source_type: sourceTypeSelect.value || null,
    k: 6,
  };

  const endpoint = state.debugMode ? "/api/retrieve" : "/api/chat";
  setStatus(state.debugMode ? "Inspecting retrieval..." : "Thinking...");
  sendButton.disabled = true;

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.debugMode ? { ...payload, query: payload.question } : payload),
    });

    if (!response.ok) {
      throw new Error(`Request failed with ${response.status}`);
    }

    const data = await response.json();
    if (state.debugMode) {
      renderAnswer("Debug mode shows raw retrieved chunks instead of a model answer.");
      renderSources((data.results || []).map((result) => ({
        index: result.index,
        title: result.metadata.title || `${result.metadata.source_type || "source"} result`,
        canonical_uri: result.metadata.canonical_uri,
        source_type: result.metadata.source_type,
        game: result.metadata.game,
        mods: (result.metadata.mods_csv || "").split("|").filter(Boolean),
        author_name: result.metadata.author_name,
        snippet: result.text,
      })));
    } else {
      renderAnswer(data.answer || "No answer returned.");
      renderSources(data.sources || []);
    }
    setStatus("Done.");
  } catch (error) {
    renderAnswer("");
    renderSources([]);
    setStatus(error.message || "Request failed.");
  } finally {
    sendButton.disabled = false;
  }
}

gameSelect.addEventListener("change", renderMods);
sendButton.addEventListener("click", submitQuestion);
debugButton.addEventListener("click", () => {
  state.debugMode = !state.debugMode;
  debugButton.textContent = state.debugMode ? "Leave Retrieval Debug" : "Inspect Retrieval";
  setStatus(state.debugMode ? "Debug mode enabled." : "");
});

loadFilters().catch((error) => {
  setStatus(error.message || "Failed to load filters.");
});
