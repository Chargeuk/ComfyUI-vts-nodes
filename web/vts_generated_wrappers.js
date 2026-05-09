import { app } from "/scripts/app.js";

const NODE_PREFIX = "VTSWrapper_";
const PREFIX_WIDGET = "vts_prefix";
const SUFFIX_LENGTH = 6;
const ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789";

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function sanitizeDisplayName(value) {
  return String(value ?? "")
    .trim()
    .replace(/\s+/g, "_");
}

function randomSuffix(length = SUFFIX_LENGTH) {
  let result = "";
  for (let index = 0; index < length; index += 1) {
    result += ALPHABET[Math.floor(Math.random() * ALPHABET.length)];
  }
  return result;
}

app.registerExtension({
  name: "VTS.GeneratedWrapperPrefixDefaults",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!String(nodeData?.name ?? "").startsWith(NODE_PREFIX)) {
      return;
    }

    const basePrefix = sanitizeDisplayName(nodeData.display_name ?? nodeData.name);
    const onNodeCreated = nodeType.prototype.onNodeCreated;

    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

      const prefixWidget = getWidget(this, PREFIX_WIDGET);
      if (!prefixWidget) {
        return;
      }

      const currentValue = String(prefixWidget.value ?? "").trim();
      if (currentValue && currentValue !== basePrefix) {
        return;
      }

      prefixWidget.value = `${basePrefix}_${randomSuffix()}`;
      this.setDirtyCanvas?.(true, true);
    };
  },
});
