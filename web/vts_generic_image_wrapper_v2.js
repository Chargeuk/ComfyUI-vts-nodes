import { app } from "/scripts/app.js";

const NODE_NAME = "VTSGenericImageWrapperV2";
const FILTER_ALL = "All";
const NO_MATCHING_NODES = "No matching nodes";
const WRAPPED_PREFIX = "wrapped__";
const RETURN_TYPES_WITH_INPUT = ["Input", "Tensor", "DiskImage"];
const RETURN_TYPES_NO_INPUT = ["Tensor", "DiskImage"];

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function getCatalog(nodeData) {
  return nodeData?.input?.required?.vts_wrapped_node_name?.[1]?.vts_node_catalog ?? {};
}

function removeWrappedInputs(node) {
  const wrappedInputs = (node.inputs ?? []).filter((input) => input.name.startsWith(WRAPPED_PREFIX));
  for (const input of wrappedInputs) {
    const slot = node.findInputSlot(input.name);
    if (slot >= 0) {
      node.removeInput(slot);
    }
  }
}

function removeWrappedWidgets(node) {
  if (!node.widgets) {
    return;
  }

  for (const widget of node.widgets.filter((item) => item.name.startsWith(WRAPPED_PREFIX))) {
    widget.onRemove?.();
  }
  node.widgets = node.widgets.filter((item) => !item.name.startsWith(WRAPPED_PREFIX));
}

function addWrappedWidget(node, name, label, legacySpec) {
  const rawType = legacySpec[0];
  const config = (legacySpec.length > 1 && typeof legacySpec[1] === "object" && legacySpec[1]) || {};
  let widget = null;

  if (Array.isArray(rawType)) {
    const values = [...rawType];
    widget = node.addWidget("combo", name, config.default ?? values[0] ?? "", () => {}, {
      values,
      serialize: true,
    });
  } else if (rawType === "STRING") {
    widget = node.addWidget("text", name, config.default ?? "", () => {}, {
      serialize: true,
      multiline: !!config.multiline,
    });
  } else if (rawType === "INT") {
    widget = node.addWidget("number", name, config.default ?? 0, () => {}, {
      min: config.min,
      max: config.max,
      step: config.step ?? 1,
      precision: 0,
      serialize: true,
    });
  } else if (rawType === "FLOAT") {
    widget = node.addWidget("number", name, config.default ?? 0, () => {}, {
      min: config.min,
      max: config.max,
      step: config.step ?? 0.01,
      precision: 4,
      serialize: true,
    });
  } else if (rawType === "BOOLEAN") {
    widget = node.addWidget("toggle", name, config.default ?? false, () => {}, {
      on: "true",
      off: "false",
      serialize: true,
    });
  }

  if (widget) {
    widget.label = label;
    return true;
  }

  return false;
}

function addWrappedInput(node, inputSpec) {
  const rawType = inputSpec.legacy_spec[0];
  const internalName = `${WRAPPED_PREFIX}${inputSpec.name}`;

  if (addWrappedWidget(node, internalName, inputSpec.name, inputSpec.legacy_spec)) {
    return;
  }

  const input = node.addInput(internalName, rawType);
  if (input) {
    input.label = inputSpec.name;
  }
}

function setComboOptions(widget, values) {
  if (!widget?.options) {
    return;
  }
  widget.options.values = [...values];
}

function wrapCallback(widget, afterChange) {
  const originalCallback = widget.callback;
  widget.callback = function () {
    const result = originalCallback?.apply(this, arguments);
    afterChange();
    return result;
  };
}

app.registerExtension({
  name: "VTS.GenericImageWrapperV2",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) {
      return;
    }

    const catalog = getCatalog(nodeData);
    const allWrappedNodeKeys = Object.keys(catalog);

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

      this.serialize_widgets = true;

      const categoryWidget = getWidget(this, "vts_category_filter");
      const packageWidget = getWidget(this, "vts_package_filter");
      const wrappedNodeWidget = getWidget(this, "vts_wrapped_node_name");
      const returnTypeWidget = getWidget(this, "vts_return_type");

      if (!categoryWidget || !packageWidget || !wrappedNodeWidget || !returnTypeWidget) {
        return;
      }

      const updateReturnTypeOptions = () => {
        const selectedWrappedNode = wrappedNodeWidget.value;
        const hasImageInput = !!catalog[selectedWrappedNode]?.has_image_input;
        const allowedReturnTypes = hasImageInput ? RETURN_TYPES_WITH_INPUT : RETURN_TYPES_NO_INPUT;
        setComboOptions(returnTypeWidget, allowedReturnTypes);
        if (!allowedReturnTypes.includes(returnTypeWidget.value)) {
          returnTypeWidget.value = hasImageInput ? "Input" : "Tensor";
        }
      };

      const syncWrappedInputs = () => {
        removeWrappedInputs(this);
        removeWrappedWidgets(this);

        const selectedWrappedNode = wrappedNodeWidget.value;
        const selectedSpec = catalog[selectedWrappedNode];
        if (!selectedSpec) {
          this.size = this.computeSize();
          this.setDirtyCanvas(true, true);
          return;
        }

        for (const inputSpec of selectedSpec.legacy_inputs ?? []) {
          addWrappedInput(this, inputSpec);
        }

        this.size = this.computeSize();
        this.setDirtyCanvas(true, true);
      };

      const updateWrappedNodeOptions = () => {
        const selectedCategory = categoryWidget.value ?? FILTER_ALL;
        const selectedPackage = packageWidget.value ?? FILTER_ALL;

        const filteredKeys = allWrappedNodeKeys.filter((key) => {
          const meta = catalog[key];
          if (!meta) {
            return false;
          }

          const categoryMatches = selectedCategory === FILTER_ALL || meta.category === selectedCategory;
          const packageMatches = selectedPackage === FILTER_ALL || meta.package === selectedPackage;
          return categoryMatches && packageMatches;
        });

        const visibleKeys = filteredKeys.length > 0 ? filteredKeys : [NO_MATCHING_NODES];
        setComboOptions(wrappedNodeWidget, visibleKeys);

        if (!filteredKeys.includes(wrappedNodeWidget.value)) {
          wrappedNodeWidget.value = visibleKeys[0];
        }

        updateReturnTypeOptions();
        syncWrappedInputs();
      };

      wrapCallback(categoryWidget, () => updateWrappedNodeOptions());
      wrapCallback(packageWidget, () => updateWrappedNodeOptions());
      wrapCallback(wrappedNodeWidget, () => {
        updateReturnTypeOptions();
        syncWrappedInputs();
      });

      setTimeout(() => {
        updateWrappedNodeOptions();
      }, 0);
    };
  },
});
