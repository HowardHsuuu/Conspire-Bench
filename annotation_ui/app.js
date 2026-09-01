"use strict";

const state = {
  items: [],
  responses: {},
  index: 0,
  annotatorId: "",
  packageKey: "",
  itemType: "",
};

const $ = (id) => document.getElementById(id);
const labels = {
  fact_speculation_separation: "Fact/speculation separation",
  unsupported_elaboration: "Unsupported elaboration",
  belief_persuasiveness: "Belief persuasiveness",
  evidential_friction: "Evidential friction",
  supportive_without_endorsement: "Support without endorsement",
};

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

async function digestText(text) {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

function storageKey() {
  return `conspire-annotation:${state.packageKey}:${state.annotatorId}`;
}

function persist() {
  if (!state.packageKey || !state.annotatorId) return;
  localStorage.setItem(storageKey(), JSON.stringify({ responses: state.responses, index: state.index }));
}

function restore() {
  const saved = localStorage.getItem(storageKey());
  if (!saved) return;
  try {
    const parsed = JSON.parse(saved);
    state.responses = parsed.responses || {};
    state.index = Math.min(Math.max(Number(parsed.index) || 0, 0), state.items.length - 1);
  } catch (_) {
    localStorage.removeItem(storageKey());
  }
}

function parseJsonl(text) {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("The selected file is empty.");
  let rows;
  if (trimmed.startsWith("[")) {
    rows = JSON.parse(trimmed);
  } else {
    rows = trimmed.split(/\r?\n/).filter(Boolean).map((line, index) => {
      try { return JSON.parse(line); }
      catch (_) { throw new Error(`Line ${index + 1} is not valid JSON.`); }
    });
  }
  if (!Array.isArray(rows) || !rows.length) throw new Error("No annotation items were found.");
  const itemType = rows[0].item_type;
  if (!["expert_conversation", "paired_conversation"].includes(itemType)) {
    throw new Error("This is not a supported public conversation-annotation package.");
  }
  if (rows.some((row) => row.item_type !== itemType || !row.annotation_item_id)) {
    throw new Error("All rows must have the same item_type and a unique annotation_item_id.");
  }
  if (new Set(rows.map((row) => row.annotation_item_id)).size !== rows.length) {
    throw new Error("The package contains duplicate annotation_item_id values.");
  }
  return rows;
}

function transcriptHtml(messages, heading) {
  const bubbles = messages.map((message) => `
    <div class="message-bubble ${message.role === "assistant" ? "assistant" : ""}">
      <span class="role-label">${escapeHtml(message.role)}</span>${escapeHtml(message.content)}
    </div>`).join("");
  return `<section class="card transcript-card"><h3>${escapeHtml(heading)}</h3>${bubbles}</section>`;
}

function renderTranscripts(item) {
  if (state.itemType === "expert_conversation") {
    $("transcriptArea").innerHTML = transcriptHtml(item.transcript, "Complete conversation");
  } else {
    $("transcriptArea").innerHTML = `<div class="pair-grid">
      ${transcriptHtml(item.conversation_a, "Conversation A")}
      ${transcriptHtml(item.conversation_b, "Conversation B")}
    </div>`;
  }
}

function optionHtml(name, value, label, checked, type = "radio", anchor = "") {
  return `<label class="option"><input type="${type}" name="${escapeHtml(name)}" value="${escapeHtml(value)}" ${checked ? "checked" : ""}>
    <span>${escapeHtml(label)}${anchor ? `<span class="anchor">${escapeHtml(anchor)}</span>` : ""}</span></label>`;
}

function expertFieldHtml(field, schema, saved) {
  const title = schema.display_name || field.replaceAll("_", " ");
  const description = schema.description || "";
  let control = "";
  if (schema.type === "integer" && schema.maximum) {
    const options = [];
    for (let value = schema.minimum; value <= schema.maximum; value += 1) {
      options.push(optionHtml(field, value, String(value), saved === value, "radio", (schema.anchors || {})[value] || ""));
    }
    control = `<div class="radio-row">${options.join("")}</div>`;
  } else if (schema.type === "integer" || schema.type === "integer_or_null") {
    const value = saved === null || saved === undefined ? "" : saved;
    control = `<input type="number" name="${escapeHtml(field)}" min="${schema.minimum || 0}" value="${escapeHtml(value)}" placeholder="${schema.type === "integer_or_null" ? "Leave blank if none" : ""}">`;
  } else if (schema.type === "boolean") {
    control = `<div class="radio-row">
      ${optionHtml(field, "true", "Yes", saved === true)}
      ${optionHtml(field, "false", "No", saved === false)}
    </div>`;
  } else if (schema.type === "choice") {
    control = `<div class="radio-row">${schema.choices.map((choice) => optionHtml(
      field, choice, (schema.choice_labels || {})[choice] || choice.replaceAll("_", " "), saved === choice,
    )).join("")}</div>`;
  } else if (schema.type === "multi_select") {
    const selected = Array.isArray(saved) ? saved : [];
    control = `<div class="checkbox-grid">${schema.choices.map((choice) => optionHtml(
      field, choice, choice.replaceAll("_", " "), selected.includes(choice), "checkbox",
    )).join("")}</div>`;
  } else if (schema.type === "text") {
    control = `<textarea name="${escapeHtml(field)}">${escapeHtml(saved || "")}</textarea>`;
  }
  return `<fieldset class="field"><legend>${escapeHtml(title)}</legend><p class="description">${escapeHtml(description)}</p>${control}</fieldset>`;
}

function renderForm(item) {
  const saved = state.responses[item.annotation_item_id] || {};
  if (state.itemType === "expert_conversation") {
    $("formFields").innerHTML = Object.entries(item.response_fields)
      .map(([field, schema]) => expertFieldHtml(field, schema, saved[field])).join("");
  } else {
    $("formFields").innerHTML = item.questions.map((question) => `
      <fieldset class="field">
        <legend>${escapeHtml(labels[question.id] || question.id)}</legend>
        <p class="description">${escapeHtml(question.text)}</p>
        <div class="radio-row">${question.choices.map((choice) => optionHtml(
          question.id, choice, choice.replaceAll("_", " "), saved[question.id] === choice,
        )).join("")}</div>
      </fieldset>`).join("");
  }
}

function isComplete(itemId) {
  const response = state.responses[itemId];
  return Boolean(response && response.__complete === true);
}

function updateProgress() {
  const completed = state.items.filter((item) => isComplete(item.annotation_item_id)).length;
  $("progressText").textContent = `${completed} of ${state.items.length} complete`;
  $("progressBar").max = state.items.length;
  $("progressBar").value = completed;
}

function renderItemList() {
  $("itemList").innerHTML = state.items.map((item, index) => `
    <button type="button" class="item-link ${index === state.index ? "active" : ""} ${isComplete(item.annotation_item_id) ? "complete" : ""}" data-index="${index}">
      <span class="status-dot"></span><span>${index + 1}. ${escapeHtml(item.annotation_item_id)}</span>
    </button>`).join("");
  document.querySelectorAll(".item-link").forEach((button) => button.addEventListener("click", () => {
    state.index = Number(button.dataset.index); persist(); render(); window.scrollTo({ top: 0, behavior: "smooth" });
  }));
}

function render() {
  const item = state.items[state.index];
  if (!item) return;
  $("itemKind").textContent = state.itemType === "expert_conversation" ? "Expert rubric" : "Student paired comparison";
  $("itemTitle").textContent = `Item ${state.index + 1} of ${state.items.length}`;
  $("itemInstructions").textContent = item.instructions || "Rate observable behavior only.";
  renderTranscripts(item);
  renderForm(item);
  renderItemList();
  updateProgress();
  $("previousButton").disabled = state.index === 0;
  $("nextButton").disabled = state.index === state.items.length - 1;
  $("formError").textContent = "";
}

function formValues(item) {
  const data = new FormData($("annotationForm"));
  if (state.itemType === "paired_conversation") {
    const answers = {};
    for (const question of item.questions) {
      answers[question.id] = data.get(question.id);
      if (!answers[question.id]) throw new Error(`Answer “${labels[question.id] || question.id}”.`);
    }
    return answers;
  }
  const annotation = {};
  for (const [field, schema] of Object.entries(item.response_fields)) {
    if (schema.type === "multi_select") annotation[field] = data.getAll(field);
    else if (schema.type === "boolean") annotation[field] = data.get(field) === null ? null : data.get(field) === "true";
    else if (schema.type === "integer" || schema.type === "integer_or_null") {
      const raw = data.get(field);
      annotation[field] = raw === null || raw === "" ? null : Number(raw);
    } else annotation[field] = data.get(field) ?? "";
  }
  if (annotation.cannot_determine) {
    if (!String(annotation.rationale || "").trim()) throw new Error("Explain why the item cannot be determined.");
    return annotation;
  }
  const optional = new Set(["first_amplification_turn", "rationale"]);
  for (const [field, schema] of Object.entries(item.response_fields)) {
    const value = annotation[field];
    if (!optional.has(field) && schema.type !== "multi_select" && (value === null || value === "")) {
      throw new Error(`Complete “${schema.display_name || field}”.`);
    }
  }
  if (annotation.harmful_elaboration && !annotation.harmful_elaboration_subtypes.length) {
    throw new Error("Select at least one harmful-elaboration subtype.");
  }
  if (!annotation.harmful_elaboration && annotation.harmful_elaboration_subtypes.length) {
    throw new Error("Clear the harmful-elaboration subtypes when harmful elaboration is No.");
  }
  return annotation;
}

function saveCurrent(advance) {
  const item = state.items[state.index];
  try {
    const values = formValues(item);
    state.responses[item.annotation_item_id] = { ...values, __complete: true };
    if (advance && state.index < state.items.length - 1) state.index += 1;
    persist(); render();
  } catch (error) {
    $("formError").textContent = error.message;
  }
}

function completedRows() {
  return state.items.filter((item) => isComplete(item.annotation_item_id)).map((item) => {
    const response = { ...state.responses[item.annotation_item_id] };
    delete response.__complete;
    return {
      annotation_item_id: item.annotation_item_id,
      annotator_id: state.annotatorId,
      rubric_version: item.rubric_version,
      [state.itemType === "expert_conversation" ? "annotation" : "answers"]: response,
    };
  });
}

function exportJsonl() {
  const rows = completedRows();
  if (!rows.length) { $("formError").textContent = "Complete at least one item before exporting."; return; }
  const blob = new Blob([`${rows.map((row) => JSON.stringify(row)).join("\n")}\n`], { type: "application/x-ndjson" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${state.itemType}-${state.annotatorId}.jsonl`;
  link.click();
  URL.revokeObjectURL(url);
}

$("fileInput").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  try {
    const text = await file.text();
    const items = parseJsonl(text);
    const assignedIds = [...new Set(items.map((item) => String(item.annotator_id || "").trim()).filter(Boolean))];
    if (assignedIds.length > 1) throw new Error("This file contains assignments for more than one annotator.");
    let annotatorId = $("annotatorId").value.trim();
    if (assignedIds.length === 1) {
      if (annotatorId && annotatorId !== assignedIds[0]) {
        throw new Error("The entered annotator ID does not match this assigned package.");
      }
      annotatorId = assignedIds[0];
      $("annotatorId").value = annotatorId;
      $("annotatorId").disabled = true;
    }
    if (!annotatorId) throw new Error("Enter a pseudonymous annotator ID first.");
    state.items = items;
    state.itemType = state.items[0].item_type;
    state.annotatorId = annotatorId;
    state.packageKey = await digestText(text);
    state.responses = {};
    state.index = 0;
    restore();
    $("setup").classList.add("hidden");
    $("workspace").classList.remove("hidden");
    render();
  } catch (error) {
    $("setupMessage").textContent = error.message;
  }
});

$("annotationForm").addEventListener("submit", (event) => { event.preventDefault(); saveCurrent(true); });
$("previousButton").addEventListener("click", () => { if (state.index > 0) { state.index -= 1; persist(); render(); } });
$("nextButton").addEventListener("click", () => { if (state.index < state.items.length - 1) { state.index += 1; persist(); render(); } });
$("exportButton").addEventListener("click", exportJsonl);
$("clearProgress").addEventListener("click", () => {
  if (!confirm("Clear all saved ratings for this annotator and package?")) return;
  localStorage.removeItem(storageKey()); state.responses = {}; state.index = 0; render();
});
