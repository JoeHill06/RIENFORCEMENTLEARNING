/* -----------------------------------------------------------------------
 * Reinforcement Learning - site script
 * One file, two entry points. `renderIndex()` runs on the landing page;
 * `renderProject()` runs on the detail page and hosts the interactive
 * Q-learning tic tac toe demo.
 * --------------------------------------------------------------------- */

/* ============================ INDEX PAGE ============================= */

async function renderIndex() {
  // Registry is the single source of truth for which projects exist.
  const registry = await (await fetch("projects/registry.json")).json();

  document.getElementById("site-title").textContent = registry.title;
  document.getElementById("site-subtitle").textContent = registry.subtitle;

  const gridEl = document.getElementById("project-grid");
  gridEl.innerHTML = registry.projects.map(projectCard).join("");
}

function projectCard(p) {
  // Each card links to project.html?id=<slug>. Reuses the sibling site's
  // `.network-card` design: hairline card, monospace "arch" line, mint
  // accuracy pill, muted techniques summary.
  const techniques = (p.bullets || []).join(" · ");
  return `
    <a class="network-card" href="project.html?id=${encodeURIComponent(p.id)}">
      <h2>${p.number}. ${escapeHtml(p.title)}</h2>
      <div class="arch">${escapeHtml(p.summary)}</div>
      <div class="desc">${escapeHtml(p.description)}</div>
      ${p.headline_stat ? `<span class="accuracy">${escapeHtml(p.headline_stat)}</span>` : ""}
      ${techniques ? `<div class="techniques">${escapeHtml(techniques)}</div>` : ""}
    </a>
  `;
}

/* ============================ PROJECT PAGE =========================== */

async function renderProject() {
  // Fall back to the first registry entry if no id was supplied, so the
  // page does not 404 on a bare visit to project.html.
  const params = new URLSearchParams(window.location.search);
  let id = params.get("id");
  if (!id) {
    const reg = await (await fetch("projects/registry.json")).json();
    id = reg.projects[0].id;
  }

  const info = await (await fetch(`projects/${id}/info.json`)).json();
  document.title = `${info.title} - Reinforcement Learning`;

  const learning = info.learning_method;
  const learningPoints = (learning?.points || [])
    .map(p => `<li>${escapeHtml(p)}</li>`).join("");

  // Notes can either be inline `paragraphs` in info.json, or a path to a
  // markdown file that we fetch and render client-side. The markdown path
  // wins if both are set.
  const notesMarkdownPath = info.notes?.markdown_path;
  const notesParagraphs = notesMarkdownPath
    ? ""
    : (info.notes?.paragraphs || [])
        .map(p => `<p>${escapeHtml(p)}</p>`).join("");

  const learned = (info.what_i_learned || [])
    .map(p => `<li>${escapeHtml(p)}</li>`).join("");

  const next = (info.things_to_try_next || [])
    .map(p => `<li>${escapeHtml(p)}</li>`).join("");

  document.getElementById("project-detail").innerHTML = `
    <section class="network-meta">
      <h1>${info.number}. ${escapeHtml(info.title)}</h1>
      <div class="arch">${escapeHtml(info.summary)}</div>
      <div class="result-summary">${escapeHtml(info.description)}</div>
    </section>

    <section class="try-it-section">
      <h2>Try it</h2>
      <div class="try-it-layout">
        <div class="try-it-left">
          <div class="ttt-wrapper">
            <div class="ttt-board" id="ttt-board"></div>
          </div>
          <div class="btn-row">
            <button class="btn-primary" id="btn-new-user">You first</button>
            <button id="btn-new-agent">Agent first</button>
          </div>
          <div class="try-it-hint">${escapeHtml(info.demo?.caption || "")}</div>
        </div>
        <div class="try-it-right">
          <div class="ttt-status" id="ttt-status">loading...</div>
          ${learning ? `
            <div class="notes-body" style="margin-top:1rem;">
              <h3>How it picks</h3>
              <p>For every empty cell, look up <code>Q(state, action)</code> in the trained table; play the cell with the highest value. Default to 0 for any pair never seen in training.</p>
              <pre><code>${escapeHtml(learning.formula || "")}</code></pre>
            </div>
          ` : ""}
        </div>
      </div>
    </section>

    ${notesMarkdownPath ? `
      <section class="notes-section">
        <h2>Notes</h2>
        <div class="notes-body" id="notes-markdown">loading notes...</div>
      </section>
    ` : (notesParagraphs ? `
      <section class="notes-section">
        <h2>Notes</h2>
        <div class="notes-body">${notesParagraphs}</div>
      </section>
    ` : "")}

    ${learning && learningPoints ? `
      <section class="notes-section">
        <h2>${escapeHtml(learning.title || "Learning method")}</h2>
        <div class="notes-body">
          ${learning.formula ? `<pre><code>${escapeHtml(learning.formula)}</code></pre>` : ""}
          <ul>${learningPoints}</ul>
        </div>
      </section>
    ` : ""}

    ${learned ? `
      <section class="notes-section">
        <h2>What I learned</h2>
        <div class="notes-body"><ul>${learned}</ul></div>
      </section>
    ` : ""}

    ${next ? `
      <section class="notes-section">
        <h2>Things to try next</h2>
        <div class="notes-body"><ul>${next}</ul></div>
      </section>
    ` : ""}
  `;

  if (notesMarkdownPath) {
    await renderMarkdownNotes(notesMarkdownPath);
  }

  if (info.demo?.type === "tic-tac-toe") {
    await initTicTacToeDemo(info.demo);
  }
}

// Fetch a markdown file, render it with marked, and run KaTeX over the
// result so the Bellman-update equation displays properly. marked/KaTeX
// are loaded via <script defer> in project.html, so they may not be ready
// when this function first runs - wait for DOMContentLoaded to be safe.
async function renderMarkdownNotes(path) {
  if (document.readyState === "loading") {
    await new Promise(r =>
      document.addEventListener("DOMContentLoaded", r, { once: true })
    );
  }
  const container = document.getElementById("notes-markdown");
  if (!container) return;

  try {
    const md = await (await fetch(path)).text();
    container.innerHTML = window.marked.parse(md);
    if (window.renderMathInElement) {
      window.renderMathInElement(container, {
        delimiters: [
          { left: "$$", right: "$$", display: true },
          { left: "$",  right: "$",  display: false },
        ],
        throwOnError: false,
      });
    }
  } catch (err) {
    container.textContent = `Failed to load notes: ${err}`;
  }
}

/* ====================== TIC TAC TOE DEMO ============================ */

// Module-level demo state; exposed on `window.ttt` for easy debugging.
const ttt = {
  qTable: null,    // map "state|r,c" -> Q-value
  board:  null,    // flat array of 9 chars: '.', 'X', 'O'
  userTurn: true,
  gameOver: false,
  agentToken: "X",
  userToken:  "O",
};
window.ttt = ttt;

async function initTicTacToeDemo(demoConfig) {
  ttt.agentToken = demoConfig.agent_token || "X";
  ttt.userToken  = demoConfig.user_token  || "O";

  // Keys look like ".X.O.....|1,2" -> 0.43; same encoding the Python
  // exporter writes.
  ttt.qTable = await (await fetch(demoConfig.q_table_path)).json();

  const boardEl = document.getElementById("ttt-board");
  boardEl.innerHTML = "";
  for (let i = 0; i < 9; i++) {
    const btn = document.createElement("button");
    btn.className = "ttt-cell";
    btn.dataset.index = i;
    btn.addEventListener("click", onCellClick);
    boardEl.appendChild(btn);
  }

  document.getElementById("btn-new-user").addEventListener("click", () => newGame(true));
  document.getElementById("btn-new-agent").addEventListener("click", () => newGame(false));

  newGame(true);
}

function newGame(userFirst) {
  ttt.board = Array(9).fill(".");
  ttt.userTurn = userFirst;
  ttt.gameOver = false;
  renderBoard();
  setStatus(userFirst ? "Your move." : "Agent thinking...");
  if (!userFirst) {
    // Small delay so the user sees the board reset before the agent moves.
    setTimeout(agentMove, 250);
  }
}

function onCellClick(e) {
  if (ttt.gameOver || !ttt.userTurn) return;
  const idx = parseInt(e.currentTarget.dataset.index, 10);
  if (ttt.board[idx] !== ".") return;

  ttt.board[idx] = ttt.userToken;
  ttt.userTurn = false;
  renderBoard();

  if (evaluateGame()) return;
  setStatus("Agent thinking...");
  setTimeout(agentMove, 250);
}

function agentMove() {
  if (ttt.gameOver) return;

  // Current board as the 9-char state string - matches the Python key format.
  const state = ttt.board.join("");

  // Enumerate legal moves and score each against the Q-table; argmax wins.
  let bestIdx = -1;
  let bestVal = -Infinity;
  for (let i = 0; i < 9; i++) {
    if (ttt.board[i] !== ".") continue;
    const r = Math.floor(i / 3);
    const c = i % 3;
    // Default 0 for unseen (state, action) pairs - optimistic init, same
    // behavior as agent.py's find_best_move.
    const v = ttt.qTable[`${state}|${r},${c}`] ?? 0;
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }

  if (bestIdx === -1) return;

  ttt.board[bestIdx] = ttt.agentToken;
  ttt.userTurn = true;
  renderBoard();
  if (evaluateGame()) return;
  setStatus("Your move.");
}

/* ------------------------- Board + win logic ------------------------- */

const WIN_LINES = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],   // rows
  [0, 3, 6], [1, 4, 7], [2, 5, 8],   // columns
  [0, 4, 8], [2, 4, 6],              // diagonals
];

function findWinner() {
  for (const line of WIN_LINES) {
    const [a, b, c] = line;
    const v = ttt.board[a];
    if (v !== "." && v === ttt.board[b] && v === ttt.board[c]) {
      return { token: v, line };
    }
  }
  return null;
}

function isDraw() {
  return ttt.board.every(c => c !== ".");
}

function evaluateGame() {
  const winner = findWinner();
  if (winner) {
    ttt.gameOver = true;
    highlightWinningLine(winner.line);
    if (winner.token === ttt.userToken) setStatus("You win!", "win");
    else                                setStatus("Agent wins.", "loss");
    return true;
  }
  if (isDraw()) {
    ttt.gameOver = true;
    setStatus("Draw.", "draw");
    return true;
  }
  return false;
}

function renderBoard() {
  const cells = document.querySelectorAll(".ttt-cell");
  cells.forEach((btn, i) => {
    const v = ttt.board[i];
    btn.textContent = v === "." ? "" : v;
    btn.classList.remove("x", "o", "winning");
    if (v === "X") btn.classList.add("x");
    if (v === "O") btn.classList.add("o");
    btn.disabled = ttt.gameOver || v !== ".";
  });
}

function highlightWinningLine(line) {
  const cells = document.querySelectorAll(".ttt-cell");
  line.forEach(i => cells[i].classList.add("winning"));
}

function setStatus(msg, cls = "") {
  const el = document.getElementById("ttt-status");
  el.textContent = msg;
  el.className = "ttt-status" + (cls ? " " + cls : "");
}

/* ---------------------------- utilities ------------------------------ */

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
