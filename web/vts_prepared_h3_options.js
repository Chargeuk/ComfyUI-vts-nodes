import { app } from "/scripts/app.js";

const NUMERIC = ["local_sparsity", "dense_first_steps", "dense_last_steps", "min_window_tokens"];
const MEMORY = ["branch_weights", "retain_buffers", "prefetch", "attention_backend"];

app.registerExtension({
  name: "VTS.PreparedH3FactoryDefaults",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "VTS_PreparedH3RuntimeOptions") return;
    const profiles = nodeData.input?.optional?.factory_defaults?.[1]?.vts_factory_profiles
      ?? nodeData.inputs?.factory_defaults?.vts_factory_profiles ?? {};
    const widget = (node, name) => node.widgets?.find((item) => item.name === name);

    const updateVisibility = (node) => {
      const profile = profiles[widget(node, "factory_defaults")?.value];
      for (const name of [...NUMERIC, ...MEMORY]) {
        const item = widget(node, name);
        if (item) item.hidden = !!profile && !(name in profile.common);
      }
      node.setDirtyCanvas?.(true, true);
    };
    const resetDefaults = (node) => {
      const profile = profiles[widget(node, "factory_defaults")?.value];
      for (const name of [...NUMERIC, ...MEMORY]) {
        const item = widget(node, name);
        if (item) item.value = profile?.common[name] ?? (NUMERIC.includes(name) ? -1 : "saved");
      }
      const advanced = widget(node, "advanced_json");
      if (advanced) advanced.value = JSON.stringify(profile?.advanced ?? {}, null, 2);
      updateVisibility(node);
    };

    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = created?.apply(this, arguments);
      const selector = widget(this, "factory_defaults");
      if (!selector) return result;
      const callback = selector.callback;
      selector.callback = (...args) => {
        callback?.apply(selector, args);
        resetDefaults(this);
      };
      resetDefaults(this);
      return result;
    };

    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const result = configured?.apply(this, arguments);
      const selector = widget(this, "factory_defaults");
      const index = this.widgets?.indexOf(selector);
      // The selector was appended after the original widgets. Keep old saved
      // overrides/inheritance intact instead of applying new factory defaults.
      if (selector && Array.isArray(info?.widgets_values) && info.widgets_values.length <= index) {
        selector.value = "Saved settings";
      }
      updateVisibility(this); // Never reset values while restoring a workflow.
      return result;
    };
  },
});
