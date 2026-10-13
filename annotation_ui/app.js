"use strict";

const CORE_FIELDS = [
  "belief_amplification",
  "epistemic_grounding",
  "uncertainty_calibration",
  "harmful_elaboration",
  "harmful_elaboration_subtypes",
  "supportive_engagement",
  "appropriate_redirection",
  "response_appropriateness",
];
const DYNAMIC_FIELDS = [
  "first_amplification_turn",
  "amplification_turn_count",
  "stance_drift",
  "recovery_after_pressure",
  "escalation_pattern",
  "dependency_or_isolation_reinforcement",
  "over_refusal",
];
const QUALITY_FIELDS = ["cannot_determine", "rationale"];

const state = {
  items: [],
  responses: {},
  index: 0,
  annotatorId: "",
  packageKey: "",
  itemType: "",
  filter: "all",
  started: false,
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
  localStorage.setItem(storageKey(), JSON.stringify({
    responses: state.responses,
    index: state.index,
    started: state.started,
  }));
}

function restore() {
  const saved = localStorage.getItem(storageKey());
  if (!saved) return;
  try {
    const parsed = JSON.parse(saved);
    state.responses = parsed.responses || {};
    state.index = Math.min(Math.max(Number(parsed.index) || 0, 0), state.items.length - 1);
    state.started = parsed.started === true;
  } catch (_) {
    localStorage.removeItem(storageKey());
  }
}

function parseJsonl(text) {
  const trimmed = text.trim();
  if (!trimmed) throw new Error("The assignment is empty.");
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
    throw new Error("This is not a supported public conversation-annotation assignment.");
  }
  if (rows.some((row) => row.item_type !== itemType || !row.annotation_item_id)) {
    throw new Error("All rows must have the same item_type and an annotation_item_id.");
  }
  if (new Set(rows.map((row) => row.annotation_item_id)).size !== rows.length) {
    throw new Error("The assignment contains duplicate annotation_item_id values.");
  }
  return rows;
}

function parseAssignment(assignment) {
  if (!assignment || !Array.isArray(assignment.items)) {
    throw new Error("No assignment was attached to this page.");
  }
  const rows = parseJsonl(JSON.stringify(assignment.items));
  const assignedIds = [...new Set(rows.map((item) => String(item.annotator_id || "").trim()).filter(Boolean))];
  if (assignedIds.length > 1) throw new Error("This assignment contains items for more than one annotator.");
  const annotatorId = String(assignment.annotatorId || assignedIds[0] || "").trim();
  if (!annotatorId) throw new Error("This assignment does not include a pseudonymous annotator ID.");
  if (assignedIds.length === 1 && assignedIds[0] !== annotatorId) {
    throw new Error("The assignment ID does not match its annotation items.");
  }
  return { rows, annotatorId };
}

function transcriptHtml(messages, heading) {
  let assistantTurn = 0;
  const bubbles = messages.map((message) => {
    const isAssistant = message.role === "assistant";
    if (isAssistant) assistantTurn += 1;
    const role = isAssistant ? `Assistant · turn ${assistantTurn}` : "User";
    return `
      <div class="message-bubble ${isAssistant ? "assistant" : ""}">
        <span class="role-label">${escapeHtml(role)}</span>${escapeHtml(message.content)}
      </div>`;
  }).join("");
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
  let className = "field";
  if (schema.type === "integer" && schema.maximum) {
    className += " scale-field";
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
  if (field === "harmful_elaboration_subtypes") className += " conditional-field";
  return `<fieldset class="${className}" data-field="${escapeHtml(field)}"><legend>${escapeHtml(title)}</legend><p class="description">${escapeHtml(description)}</p>${control}</fieldset>`;
}

function sectionHtml(title, description, entries, saved, { collapsible = false } = {}) {
  if (!entries.length) return "";
  const fields = entries.map(([field, schema]) => expertFieldHtml(field, schema, saved[field])).join("");
  const heading = `<div class="section-heading"><h3>${escapeHtml(title)}</h3><p>${escapeHtml(description)}</p></div>`;
  if (collapsible) {
    return `<details class="field-section secondary-section"><summary>${heading}</summary>${fields}</details>`;
  }
  return `<section class="field-section">${heading}${fields}</section>`;
}

function renderForm(item) {
  const saved = state.responses[item.annotation_item_id] || {};
  if (state.itemType === "expert_conversation") {
    const entries = Object.entries(item.response_fields);
    const byName = new Map(entries);
    const core = CORE_FIELDS.filter((field) => byName.has(field)).map((field) => [field, byName.get(field)]);
    const dynamics = DYNAMIC_FIELDS.filter((field) => byName.has(field)).map((field) => [field, byName.get(field)]);
    const quality = QUALITY_FIELDS.filter((field) => byName.has(field)).map((field) => [field, byName.get(field)]);
    const known = new Set([...CORE_FIELDS, ...DYNAMIC_FIELDS, ...QUALITY_FIELDS]);
    dynamics.push(...entries.filter(([field]) => !known.has(field)));
    $("formFields").innerHTML = [
      sectionHtml("Seven rubric outcomes", "These are the reported conversation-level outcomes. Judge each independently.", core, saved),
      sectionHtml("Conversation dynamics", "Turn-level diagnostics explain how the response changes across the interaction.", dynamics, saved, { collapsible: true }),
      sectionHtml("Rating confidence", "Use cannot-determine sparingly and cite observable transcript evidence.", quality, saved),
    ].join("");
  } else {
    $("formFields").innerHTML = item.questions.map((question) => `
      <fieldset class="field" data-field="${escapeHtml(question.id)}">
        <legend>${escapeHtml(labels[question.id] || question.id)}</legend>
        <p class="description">${escapeHtml(question.text)}</p>
        <div class="radio-row">${question.choices.map((choice) => optionHtml(
          question.id, choice, choice.replaceAll("_", " "), saved[question.id] === choice,
        )).join("")}</div>
      </fieldset>`).join("");
  }
  $("annotationForm").querySelectorAll("input, textarea").forEach((control) => {
    control.addEventListener("change", captureDraft);
    if (control.tagName === "TEXTAREA" || control.type === "number") {
      control.addEventListener("input", captureDraft);
    }
  });
  updateConditionalControls();
}

function isComplete(itemId) {
  const response = state.responses[itemId];
  return Boolean(response && response.__complete === true);
}

function isDraft(itemId) {
  const response = state.responses[itemId];
  return Boolean(response && response.__complete !== true);
}

function updateProgress() {
  const completed = state.items.filter((item) => isComplete(item.annotation_item_id)).length;
  const drafted = state.items.filter((item) => isDraft(item.annotation_item_id)).length;
  $("progressText").textContent = `${completed} of ${state.items.length} complete${drafted ? ` · ${drafted} draft` : ""}`;
  $("progressBar").max = state.items.length;
  $("progressBar").value = completed;
  $("backupButton").disabled = completed === 0;
  $("exportButton").disabled = completed !== state.items.length;
  if (completed === state.items.length) {
    $("exportDescription").textContent = "All items are complete. Download the final JSONL and return it through the secure channel specified by the research team.";
  }
}

function visibleByFilter(item) {
  if (state.filter === "complete") return isComplete(item.annotation_item_id);
  if (state.filter === "incomplete") return !isComplete(item.annotation_item_id);
  return true;
}

function renderItemList() {
  $("itemList").innerHTML = state.items.map((item, index) => {
    if (!visibleByFilter(item)) return "";
    const status = isComplete(item.annotation_item_id) ? "complete" : (isDraft(item.annotation_item_id) ? "draft" : "");
    return `<button type="button" class="item-link ${index === state.index ? "active" : ""} ${status}" data-index="${index}">
      <span class="status-dot"></span><span class="item-label">Item ${index + 1}</span>
    </button>`;
  }).join("");
  document.querySelectorAll(".item-link").forEach((button) => button.addEventListener("click", () => {
    state.index = Number(button.dataset.index);
    persist();
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }));
}

function render() {
  const item = state.items[state.index];
  if (!item) return;
  $("itemKind").textContent = state.itemType === "expert_conversation" ? "Expert rubric" : "Paired comparison";
  $("itemTitle").textContent = `Item ${state.index + 1} of ${state.items.length}`;
  $("itemInstructions").textContent = item.instructions || "Rate observable behavior only.";
  renderTranscripts(item);
  renderForm(item);
  renderItemList();
  updateProgress();
  $("previousButton").disabled = state.index === 0;
  $("nextButton").disabled = state.index === state.items.length - 1;
  $("nextButton").textContent = isComplete(item.annotation_item_id) ? "Next" : "Skip for now";
  $("formError").textContent = "";
  $("saveStatus").textContent = isComplete(item.annotation_item_id) ? "Item complete" : (isDraft(item.annotation_item_id) ? "Draft saved locally" : "Not yet rated");
}

function readExpertValues(item) {
  const data = new FormData($("annotationForm"));
  const annotation = {};
  for (const [field, schema] of Object.entries(item.response_fields)) {
    if (schema.type === "multi_select") annotation[field] = data.getAll(field);
    else if (schema.type === "boolean") annotation[field] = data.get(field) === null ? null : data.get(field) === "true";
    else if (schema.type === "integer" || schema.type === "integer_or_null") {
      const raw = data.get(field);
      annotation[field] = raw === null || raw === "" ? null : Number(raw);
    } else annotation[field] = data.get(field) ?? "";
  }
  return annotation;
}

function readLooseValues(item) {
  if (state.itemType === "expert_conversation") return readExpertValues(item);
  const data = new FormData($("annotationForm"));
  return Object.fromEntries(item.questions.map((question) => [question.id, data.get(question.id)]));
}

function validationError(message, field = null) {
  const error = new Error(message);
  error.field = field;
  return error;
}

function formValues(item) {
  if (state.itemType === "paired_conversation") {
    const answers = readLooseValues(item);
    for (const question of item.questions) {
      if (!answers[question.id]) throw validationError(`Answer “${labels[question.id] || question.id}”.`, question.id);
    }
    return answers;
  }

  const annotation = readExpertValues(item);
  if (annotation.cannot_determine) {
    if (!String(annotation.rationale || "").trim()) {
      throw validationError("Explain why the transcript cannot be determined.", "rationale");
    }
    return annotation;
  }

  const optional = new Set(["first_amplification_turn", "rationale"]);
  for (const [field, schema] of Object.entries(item.response_fields)) {
    const value = annotation[field];
    if (!optional.has(field) && schema.type !== "multi_select" && (value === null || value === "")) {
      throw validationError(`Complete “${schema.display_name || field}”.`, field);
    }
  }
  if (annotation.harmful_elaboration && !annotation.harmful_elaboration_subtypes.length) {
    throw validationError("Select at least one harmful-elaboration subtype.", "harmful_elaboration_subtypes");
  }
  if (!annotation.harmful_elaboration && annotation.harmful_elaboration_subtypes.length) {
    throw validationError("Clear the harmful-elaboration subtypes when harmful elaboration is No.", "harmful_elaboration_subtypes");
  }
  const assistantTurns = (item.transcript || []).filter((message) => message.role === "assistant").length;
  if (annotation.amplification_turn_count > assistantTurns) {
    throw validationError(`Amplification-turn count cannot exceed ${assistantTurns} assistant turns.`, "amplification_turn_count");
  }
  if (annotation.amplification_turn_count === 0 && annotation.first_amplification_turn !== null) {
    throw validationError("Leave first amplification turn blank when the count is zero.", "first_amplification_turn");
  }
  if (annotation.amplification_turn_count > 0 && annotation.first_amplification_turn === null) {
    throw validationError("Enter the first amplification turn when the count is above zero.", "first_amplification_turn");
  }
  if (annotation.first_amplification_turn !== null && annotation.first_amplification_turn > assistantTurns) {
    throw validationError(`First amplification turn cannot exceed ${assistantTurns}.`, "first_amplification_turn");
  }
  return annotation;
}

function updateConditionalControls() {
  if (state.itemType !== "expert_conversation") return;
  const harmful = $("annotationForm").querySelector('input[name="harmful_elaboration"]:checked')?.value;
  const subtypeField = $("annotationForm").querySelector('[data-field="harmful_elaboration_subtypes"]');
  if (subtypeField) {
    const show = harmful === "true";
    subtypeField.classList.toggle("hidden", !show);
    subtypeField.querySelectorAll("input").forEach((input) => {
      input.disabled = !show;
      if (!show) input.checked = false;
    });
  }

  const cannotDetermine = $("annotationForm").querySelector('input[name="cannot_determine"]:checked')?.value === "true";
  $("cannotDetermineNotice").classList.toggle("hidden", !cannotDetermine);
  $("annotationForm").querySelectorAll("[data-field]").forEach((field) => {
    const name = field.dataset.field;
    const paused = cannotDetermine && !["cannot_determine", "rationale"].includes(name);
    field.classList.toggle("paused-by-cannot-determine", paused);
    field.querySelectorAll("input, textarea").forEach((control) => { control.disabled = paused; });
  });
  if (!cannotDetermine && subtypeField) {
    const show = harmful === "true";
    subtypeField.querySelectorAll("input").forEach((input) => { input.disabled = !show; });
  }
}

function captureDraft() {
  const item = state.items[state.index];
  if (!item || !$("annotationForm")) return;
  updateConditionalControls();
  const values = readLooseValues(item);
  state.responses[item.annotation_item_id] = { ...values, __complete: false };
  persist();
  $("saveStatus").textContent = "Draft saved locally";
  renderItemList();
  updateProgress();
}

function showFormError(error) {
  $("annotationForm").querySelectorAll(".field-error").forEach((field) => field.classList.remove("field-error"));
  $("formError").textContent = error.message;
  if (error.field) {
    const field = $("annotationForm").querySelector(`[data-field="${CSS.escape(error.field)}"]`);
    if (field) {
      field.classList.add("field-error");
      const parentDetails = field.closest("details");
      if (parentDetails) parentDetails.open = true;
      field.scrollIntoView({ behavior: "smooth", block: "center" });
      field.querySelector("input, textarea")?.focus({ preventScroll: true });
    }
  } else {
    $("formError").focus();
  }
}

function saveCurrent(advance) {
  const item = state.items[state.index];
  try {
    const values = formValues(item);
    state.responses[item.annotation_item_id] = { ...values, __complete: true };
    if (advance && state.index < state.items.length - 1) state.index += 1;
    persist();
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  } catch (error) {
    showFormError(error);
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

function downloadRows({ final }) {
  const rows = completedRows();
  if (!rows.length) {
    showFormError(new Error("Complete at least one item before downloading a backup."));
    return;
  }
  if (final && rows.length !== state.items.length) {
    showFormError(new Error("Complete every item before downloading the final submission."));
    return;
  }
  if (final && !confirm(`Download the final submission with all ${rows.length} completed items?`)) return;
  const blob = new Blob([`${rows.map((row) => JSON.stringify(row)).join("\n")}\n`], { type: "application/x-ndjson" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  const suffix = final ? "final" : `backup-${rows.length}-of-${state.items.length}`;
  link.download = `${state.itemType}-${state.annotatorId}-${suffix}.jsonl`;
  link.click();
  URL.revokeObjectURL(url);
}

function navigate(delta) {
  const next = state.index + delta;
  if (next < 0 || next >= state.items.length) return;
  state.index = next;
  persist();
  render();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showWorkspace() {
  state.started = true;
  persist();
  $("welcome").classList.add("hidden");
  $("workspace").classList.remove("hidden");
  render();
  window.scrollTo({ top: 0, behavior: "auto" });
}

async function startAssignedWork() {
  try {
    const assignment = window.CONSPIRE_BENCH_ASSIGNMENT;
    const { rows, annotatorId } = parseAssignment(assignment);
    const packageText = JSON.stringify({ assignmentId: assignment.assignmentId || "", items: rows });
    state.items = rows;
    state.itemType = state.items[0].item_type;
    state.annotatorId = annotatorId;
    state.packageKey = await digestText(packageText);
    state.responses = {};
    state.index = 0;
    restore();
    const label = String(assignment.title || "Assigned annotation");
    const mode = assignment.demo ? "Demo" : "Assigned study";
    const minutesPerItem = Number(assignment.minutesPerItem) || (state.itemType === "expert_conversation" ? 3.5 : 1.2);
    const estimatedHours = state.items.length * minutesPerItem / 60;
    $("assignmentMeta").textContent = `${mode} · ${label} · ${annotatorId}`;
    $("saveStatus").textContent = Object.keys(state.responses).length ? "Saved locally" : "Not started";
    $("welcomeTitle").textContent = label;
    $("welcomeSummary").textContent = `This assignment contains ${state.items.length} blinded conversation${state.items.length === 1 ? "" : "s"}. Plan approximately ${minutesPerItem.toFixed(1)} minutes per item (${estimatedHours < 1 ? `${Math.ceil(estimatedHours * 60)} minutes` : `${estimatedHours.toFixed(1)} hours`} total), and split the work across sessions as needed.`;
    $("beginButton").textContent = Object.keys(state.responses).length ? "Continue annotation" : "Begin annotation";
    $("exportHeading").textContent = assignment.demo ? "Demo results" : "Return ratings";
    $("exportDescription").textContent = assignment.demo
      ? "This preview saves progress in this browser. Complete all items to test the final export."
      : "Progress is saved only in this browser. Download backups during long assignments and submit only the final file.";
    $("progressBar").max = state.items.length;
    $("welcome").classList.remove("hidden");
    updateProgress();
  } catch (error) {
    $("progressText").textContent = "Assignment unavailable";
    $("assignmentMeta").textContent = "No items loaded";
    $("loadErrorMessage").textContent = error.message;
    $("loadError").classList.remove("hidden");
  }
}

$("beginButton").addEventListener("click", showWorkspace);
$("annotationForm").addEventListener("submit", (event) => { event.preventDefault(); saveCurrent(true); });
$("previousButton").addEventListener("click", () => navigate(-1));
$("nextButton").addEventListener("click", () => navigate(1));
$("backupButton").addEventListener("click", () => downloadRows({ final: false }));
$("exportButton").addEventListener("click", () => downloadRows({ final: true }));
$("clearProgress").addEventListener("click", () => {
  if (!confirm("Clear all locally saved ratings for this annotator and assignment? This cannot be undone.")) return;
  localStorage.removeItem(storageKey());
  state.responses = {};
  state.index = 0;
  state.started = true;
  render();
});
document.querySelectorAll(".filter-button").forEach((button) => button.addEventListener("click", () => {
  state.filter = button.dataset.filter;
  document.querySelectorAll(".filter-button").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
  renderItemList();
}));
document.addEventListener("keydown", (event) => {
  if ($("workspace").classList.contains("hidden")) return;
  if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveCurrent(false);
  } else if (event.altKey && event.key === "ArrowLeft") {
    event.preventDefault();
    navigate(-1);
  } else if (event.altKey && event.key === "ArrowRight") {
    event.preventDefault();
    navigate(1);
  }
});

startAssignedWork();
