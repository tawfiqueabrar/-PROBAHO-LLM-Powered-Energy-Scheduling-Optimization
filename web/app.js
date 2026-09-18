const $ = (selector) => document.querySelector(selector);
const hourRows = $("#hour-rows");
const notes = $("#notes");
const message = $("#form-message");

function tariff(hour) { return hour < 6 ? 5 : hour >= 18 && hour <= 21 ? 20 : 10; }
function addHourRows() {
  hourRows.innerHTML = Array.from({ length: 24 }, (_, hour) => `
    <tr><td>${String(hour).padStart(2, "0")}:00</td>
    <td><input class="hour-input demand" type="number" min="0" value="100" aria-label="Demand hour ${hour}"></td>
    <td><input class="hour-input solar" type="number" min="0" value="${hour >= 9 && hour <= 15 ? 35 : 0}" aria-label="Solar hour ${hour}"></td>
    <td><input class="hour-input tariff" type="number" min="0" step="0.1" value="${tariff(hour)}" aria-label="Tariff hour ${hour}"></td></tr>`).join("");
}

function addNote(value = "") {
  const count = notes.querySelectorAll(".operator-note").length;
  if (count >= 3) return;
  const label = document.createElement("label");
  label.className = "note-label";
  label.innerHTML = `<span>${String(count + 1).padStart(2, "0")}</span><textarea class="operator-note" rows="3" required placeholder="Describe a temporary energy operating condition">${value}</textarea>`;
  notes.append(label);
  $("#add-note").hidden = count + 1 >= 3;
}

function numeric(selector) { return Number($(selector).value); }
function payload() {
  const rows = [...hourRows.querySelectorAll("tr")];
  return {
    scenario_id: $("#scenario-id").value.trim(),
    operator_notes: [...notes.querySelectorAll(".operator-note")].map((item) => item.value.trim()).filter(Boolean),
    hours: rows.map((row, hour) => ({
      hour,
      demand_kwh: Number(row.querySelector(".demand").value),
      solar_kwh: Number(row.querySelector(".solar").value),
      tariff_bdt_per_kwh: Number(row.querySelector(".tariff").value),
    })),
    battery: {
      capacity_kwh: numeric("#capacity"), initial_energy_kwh: numeric("#initial"),
      minimum_energy_kwh: numeric("#minimum"), max_charge_kwh_per_hour: numeric("#max-charge"),
      max_discharge_kwh_per_hour: numeric("#max-discharge"),
    },
  };
}

function format(value) { return new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 }).format(value); }
function render(result) {
  $("#results-empty").hidden = true;
  $("#results-content").hidden = false;
  $("#result-scenario").textContent = result.scenario_id;
  $("#metric-cost").textContent = format(result.total_cost_bdt);
  $("#metric-grid").textContent = format(result.total_grid_kwh);
  $("#metric-peak").textContent = format(result.peak_grid_kwh);
  $("#summary").textContent = result.plan_summary;

  const directives = $("#directive-list"); directives.replaceChildren();
  result.directive_interpretation.forEach((entry) => {
    const item = document.createElement("div"); item.className = "directive";
    const type = document.createElement("div"); type.className = "type"; type.textContent = entry.directive_type.replaceAll("_", " ");
    const detail = document.createElement("p"); detail.textContent = entry.explanation || "No operational adjustment.";
    item.append(type, detail); directives.append(item);
  });

  const peak = Math.max(...result.hourly_plan.map((row) => row.grid_kwh), 1);
  const chart = $("#dispatch-chart"); chart.replaceChildren();
  result.hourly_plan.forEach((row) => {
    const bar = document.createElement("div");
    bar.className = `bar ${row.battery_action === "idle" ? "" : row.battery_action}`;
    bar.style.setProperty("--bar", `${Math.max(3, row.grid_kwh / peak * 100)}%`);
    bar.dataset.hour = row.hour % 3 === 0 ? row.hour : "";
    bar.title = `Hour ${row.hour}: ${format(row.grid_kwh)} kWh grid, ${row.battery_action}`;
    chart.append(bar);
  });

  const planRows = $("#plan-rows"); planRows.replaceChildren();
  result.hourly_plan.forEach((row) => {
    const tr = document.createElement("tr");
    [row.hour, format(row.grid_kwh), format(row.solar_used_kwh), row.battery_action, format(row.battery_kwh), format(row.battery_energy_after_kwh)].forEach((value) => {
      const td = document.createElement("td"); td.textContent = value; tr.append(td);
    });
    planRows.append(tr);
  });
}

$("#scenario-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const data = payload();
  if (!data.scenario_id || !data.operator_notes.length) { message.textContent = "Add a scenario ID and at least one operator note."; return; }
  const button = $("#optimize"); button.disabled = true; button.querySelector("span").textContent = "Optimizing…"; message.textContent = "Interpreting directives and calculating the lowest-cost plan…";
  try {
    const response = await fetch("/optimize-energy", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || "The schedule could not be optimized.");
    render(body); message.textContent = "Schedule optimized successfully.";
    $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) { message.textContent = error.message || "Unable to reach the optimization service."; }
  finally { button.disabled = false; button.querySelector("span").textContent = "Optimize schedule"; }
});

$("#add-note").addEventListener("click", () => addNote());
$("#load-demo").addEventListener("click", () => {
  $("#scenario-id").value = "CAMPUS-DEMO-001";
  notes.innerHTML = ""; addNote("Do not charge the battery from 2 AM until 4 AM for maintenance.");
  addNote("Keep at least 60 kWh in reserve from 6 PM until 9 PM for emergency operations.");
  addHourRows(); message.textContent = "Demo scenario loaded. Edit any values, then optimize.";
});
addHourRows();
