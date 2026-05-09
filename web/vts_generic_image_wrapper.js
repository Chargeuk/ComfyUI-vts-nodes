import { app } from "/scripts/app.js";

const NODE_NAME = "VTSGenericImageWrapper";
const FILTER_ALL = "All";
const NO_MATCHING_NODES = "No matching nodes";
const ANCHOR_WIDGET_NAME = "wrapped_node.__vts_dynamic_anchor";
const RETURN_TYPES_WITH_INPUT = ["Input", "Tensor", "DiskImage"];
const RETURN_TYPES_NO_INPUT = ["Tensor", "DiskImage"];

function getWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function getWrappedNodeMeta(nodeData) {
  const legacySpec = nodeData?.input?.required?.wrapped_node;
  if (Array.isArray(legacySpec) && legacySpec[1]?.vts_node_meta) {
    return legacySpec[1].vts_node_meta;
  }

  return nodeData?.inputs?.wrapped_node?.vts_node_meta ?? {};
}

function hideAnchorWidgets(node) {
  for (const widget of node.widgets ?? []) {
    if (widget.name === ANCHOR_WIDGET_NAME) {
      widget.hidden = true;
      widget.serialize = false;
    }
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
  name: "VTS.GenericImageWrapperFilters",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_NAME) {
      return;
    }

    const wrappedNodeMeta = getWrappedNodeMeta(nodeData);
    const allWrappedNodeKeys = Object.keys(wrappedNodeMeta);

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

      const categoryWidget = getWidget(this, "category_filter");
      const packageWidget = getWidget(this, "package_filter");
      const wrappedNodeWidget = getWidget(this, "wrapped_node");
      const returnTypeWidget = getWidget(this, "return_type");

      if (!categoryWidget || !packageWidget || !wrappedNodeWidget || !returnTypeWidget) {
        return;
      }

      const updateReturnTypeOptions = () => {
        const selectedWrappedNode = wrappedNodeWidget.value;
        const imageInputCount = wrappedNodeMeta[selectedWrappedNode]?.image_input_count ?? 0;
        const allowedReturnTypes = imageInputCount === 1 ? RETURN_TYPES_WITH_INPUT : RETURN_TYPES_NO_INPUT;

        setComboOptions(returnTypeWidget, allowedReturnTypes);
        if (!allowedReturnTypes.includes(returnTypeWidget.value)) {
          returnTypeWidget.value = imageInputCount === 1 ? "Input" : "Tensor";
        }
      };

      const updateWrappedNodeOptions = () => {
        const selectedCategory = categoryWidget.value ?? FILTER_ALL;
        const selectedPackage = packageWidget.value ?? FILTER_ALL;

        const filteredKeys = allWrappedNodeKeys.filter((key) => {
          const meta = wrappedNodeMeta[key];
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
        } else {
          updateReturnTypeOptions();
          hideAnchorWidgets(this);
          this.setDirtyCanvas(true, true);
        }
      };

      wrapCallback(categoryWidget, () => updateWrappedNodeOptions());
      wrapCallback(packageWidget, () => updateWrappedNodeOptions());
      wrapCallback(wrappedNodeWidget, () => {
        updateReturnTypeOptions();
        hideAnchorWidgets(this);
        this.setDirtyCanvas(true, true);
      });

      setTimeout(() => {
        updateWrappedNodeOptions();
        hideAnchorWidgets(this);
        this.setDirtyCanvas(true, true);
      }, 0);
    };
  },
});
