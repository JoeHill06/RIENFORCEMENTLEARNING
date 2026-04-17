/* -----------------------------------------------------------------------
 * Reinforcement Learning - site script
 * One file used by both pages. `renderIndex()` runs on the project list,
 * `renderProject()` runs on the detail page (which hosts the interactive
 * Q-learning tic tac toe demo).
 * --------------------------------------------------------------------- */

/* ============================ INDEX PAGE ============================= */

async function renderIndex() {
  // The registry is the single source of truth for which projects exist.
  const res = await fetch("projects/registry.json");
  const registry = await res.json();

  document.getElementById("site-title").textContent = registry.title;
  document.getElementById("site-subtitle").textContent = registry.subtitle;

  const listEl = document.getElementById("project-list");
  listEl.innerHTML = registry.projects.map(projectCard).join("");
}

function projectCard(p) {
  // One card per project - title links to the detail page with ?id=<slug>.
  const bullets = (p.bullets || []).map(b => `<li>${escapeHtml(b)}</li>`).join("");
  return `
    <article class="project-card">
      <h2>
        <a href="project.html?id=${encodeURIComponent(p.id)}">
          ${p.number}. ${escapeHtml(p.title)}
        </a>
      </h2>
      <p class="summary">${escapeHtml(p.summary)}</p>
      <p class="description">${escapeHtml(p.description)}</p>
      ${p.headline_stat ? `<div class="stat-badge">${escapeHtml(p.headline_stat)}</div>` : ""}
      ${bullets ? `<ul class="bullets">${bullets}</ul>` : ""}
    </article>
  `;
}

/* ============================ PROJECT PAGE =========================== */

async function renderProject() {
  // Fall back to the first registry entry if no id was supplied, so the
  // page doesn't 404 on a bare visit to project.html.
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

  const notes = (info.notes?.paragraphs || [])
    .map(p => `<p>${escapeHtml(p)}</p>`).join("");

  const learned = (info.what_i_learned || [])
    .map(p => `<li>${escapeHtml(p)}</li>`).join("");

  const next = (info.things_to_try_next || [])
    .map(p => `<li>${escapeHtml(p)}</li>`).join("");

  document.getElementById("project-detail").innerHTML = `
    <h1>${info.number}. ${escapeHtml(info.title)}</h1>
    <p class="summary">${escapeHtml(info.summary)}</p>
    <p class="description">${escapeHtml(info.description)}</p>

    <section class="try-it">
      <div class="try-it-header">
        <h3>Try it</h3>
      </div>
      <div class="demo-wrapper" id="demo-wrapper">
        <div id="status">loading...</div>
        <div class="board" id="board"></div>
        <div class="controls">
          <button class="btn primary" id="btn-new-user">New game (you first)</button>
          <button class="btn" id="btn-new-agent">New game (agent first)</button>
        </div>
        <p class="try-it-caption">${escapeHtml(info.demo?.caption || "")}</p>
      </div>
    </section>

    <h3 class="section-heading">Notes</h3>
    <div class="notes">${notes}</div>

    ${learning ? `
      <h3 class="section-heading">${escapeHtml(learning.title || "Learning method")}</h3>
      <div class="learning-method">
        ${learning.formula ? `<div class="formula">${escapeHtml(learning.formula)}</div>` : ""}
        <ul>${learningPoints}</ul>
      </div>
    ` : ""}

    ${learned ? `
      <h3 class="section-heading">What I learned</h3>
      <div class="list-block"><ul>${learned}</ul></div>
    ` : ""}

    ${next ? `
      <h3 class="section-heading">Things to try next</h3>
      <div class="list-block"><ul>${next}</ul></div>
    ` : ""}
  `;

  if (info.demo?.type === "tic-tac-toe") {
    await initTicTacToeDemo(info.demo);
  }
}

/* ====================== TIC TAC TOE DEMO ============================ */

// Module-level state for the demo. Lives on `window.ttt` for easy debug.
const ttt = {
  qTable: null,           // Map of "state|r,c" -> Q-value
  board:  null,           // array of 9 chars: '.', 'X', 'O'
  userTurn: true,
  gameOver: false,
  agentToken: "X",
  userToken:  "O",
};
window.ttt = ttt;

async function initTicTacToeDemo(demoConfig) {
  ttt.agentToken = demoConfig.agent_token || "X";
  ttt.userToken  = demoConfig.user_token  || "O";

  // Load the exported Q-table. Keys look like ".X.O.....|1,2" -> 0.43.
  const raw = await (await fetch(demoConfig.q_table_path)).json();
  ttt.qTable = raw;

  // Render the 9 empty cells once; handlers are attached via delegation.
  const boardEl = document.getElementById("board");
  boardEl.innerHTML = "";
  for (let i = 0; i < 9; i++) {
    const btn = document.createElement("button");
    btn.className = "cell";
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

  if (evaluateGame()) return;   // user might have just won (unlikely from 1 move)
  setStatus("Agent thinking...");
  setTimeout(agentMove, 250);   // tiny pause so the move feels intentional
}

function agentMove() {
  if (ttt.gameOver) return;

  // Encode current board as 9-char state string; exactly the format used
  // by the Python exporter (row by row, '.', 'X', 'O').
  const state = ttt.board.join("");

  // Enumerate legal moves (empty cells) and score each against the Q-table.
  let bestIdx = -1;
  let bestVal = -Infinity;
  for (let i = 0; i < 9; i++) {
    if (ttt.board[i] !== ".") continue;
    const r = Math.floor(i / 3);
    const c = i % 3;
    const key = `${state}|${r},${c}`;
    // Default to 0 for unseen (state, action) pairs - matches the agent's
    // optimistic-init behavior in agent.py's find_best_move.
    const v = ttt.qTable[key] ?? 0;
    if (v > bestVal) {
      bestVal = v;
      bestIdx = i;
    }
  }

  if (bestIdx === -1) return;   // no legal moves; evaluateGame will catch it

  ttt.board[bestIdx] = ttt.agentToken;
  ttt.userTurn = true;
  renderBoard();
  if (evaluateGame()) return;
  setStatus("Your move.");
}

/* ------------------------- Board + win logic ------------------------- */

// Eight winning index triples on a flat 9-element board.
const WIN_LINES = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],      // rows
  [0, 3, 6], [1, 4, 7], [2, 5, 8],      // columns
  [0, 4, 8], [2, 4, 6],                 // diagonals
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

// Returns true when the game has ended (and updates UI accordingly).
function evaluateGame() {
  const winner = findWinner();
  if (winner) {
    ttt.gameOver = true;
    highlightWinningLine(winner.line);
    if (winner.token === ttt.userToken) {
      setStatus("You win!", "win");
    } else {
      setStatus("Agent wins.", "loss");
    }
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
  const cells = document.querySelectorAll(".cell");
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
  const cells = document.querySelectorAll(".cell");
  line.forEach(i => cells[i].classList.add("winning"));
}

function setStatus(msg, cls = "") {
  const el = document.getElementById("status");
  el.textContent = msg;
  el.className = cls;
}

/* ---------------------------- utilities ------------------------------ */

function escapeHtml(s) {
  // Safe-by-default: data comes from JSON files but we never trust it.
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
