import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";
import test from "node:test";

const profiles = {
  "VTS Hybrid": {
    common: { local_sparsity: 0.7, dense_first_steps: 1, dense_last_steps: 1,
      min_window_tokens: 4096, branch_weights: "auto", retain_buffers: "auto",
      prefetch: "auto", attention_backend: "grouped" },
    advanced: { hybrid: { dense_blocks: "", verbose: true } },
  },
  "H3 SLA": {
    common: { local_sparsity: 0.8, dense_first_steps: 0, dense_last_steps: 0, min_window_tokens: 12228 },
    advanced: { sla: { block_size: 32, dense_steps: "1" } },
  },
};
let extension;
vm.runInNewContext(readFileSync(new URL("../web/vts_prepared_h3_options.js", import.meta.url), "utf8")
  .replace('import { app } from "/scripts/app.js";', ""),
  { app: { registerExtension(value) { extension = value; } } });

class Node {
  constructor() {
    this.widgets = [...Object.keys(profiles["VTS Hybrid"].common), "advanced_json", "factory_defaults"]
      .map(name => ({ name, value: name === "factory_defaults" ? "VTS Hybrid" : null }));
  }
  onConfigure(info) { info.widgets_values.forEach((value, i) => { this.widgets[i].value = value; }); }
  setDirtyCanvas() {}
}
extension.beforeRegisterNodeDef(Node, {
  name: "VTS_PreparedH3RuntimeOptions",
  input: { optional: { factory_defaults: [Object.keys(profiles), { vts_factory_profiles: profiles }] } },
});
const get = (node, name) => node.widgets.find(w => w.name === name);

test("new nodes show actual hybrid factory values", () => {
  const node = new Node(); node.onNodeCreated();
  assert.equal(get(node, "local_sparsity").value, 0.7);
  assert.equal(get(node, "dense_first_steps").value, 1);
  assert.equal(get(node, "retain_buffers").value, "auto");
  assert.equal(JSON.parse(get(node, "advanced_json").value).hybrid.verbose, true);
});

test("changing profile resets values and hides irrelevant controls", () => {
  const node = new Node(); node.onNodeCreated();
  const profile = get(node, "factory_defaults");
  profile.value = "H3 SLA"; profile.callback(profile.value);
  assert.equal(get(node, "local_sparsity").value, 0.8);
  assert.equal(get(node, "min_window_tokens").value, 12228);
  assert.equal(get(node, "dense_last_steps").value, 0);
  assert.equal(get(node, "branch_weights").hidden, true);
  assert.equal(JSON.parse(get(node, "advanced_json").value).sla.dense_steps, "1");
  profile.value = "Saved settings"; profile.callback(profile.value);
  assert.equal(get(node, "local_sparsity").value, -1);
  assert.equal(get(node, "branch_weights").value, "saved");
  assert.equal(get(node, "branch_weights").hidden, false);
});

test("restoring a workflow preserves customized values", () => {
  const node = new Node(); node.onNodeCreated();
  const values = node.widgets.map(w => w.value);
  values[0] = 0.9; values[1] = 0;
  node.onConfigure({ widgets_values: values });
  assert.equal(get(node, "local_sparsity").value, 0.9);
  assert.equal(get(node, "dense_first_steps").value, 0);
  assert.equal(get(node, "factory_defaults").value, "VTS Hybrid");
});

test("old workflows retain inheritance rather than silently applying factories", () => {
  const node = new Node(); node.onNodeCreated();
  const oldValues = [-1, 0, 0, -1, "saved", "saved", "saved", "saved", "{}"];
  node.onConfigure({ widgets_values: oldValues });
  assert.equal(get(node, "factory_defaults").value, "Saved settings");
  assert.equal(get(node, "local_sparsity").value, -1);
  assert.equal(get(node, "dense_first_steps").value, 0);
});
