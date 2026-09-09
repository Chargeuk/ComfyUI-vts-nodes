import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "VTS.MerserkSizing",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VTS Merserk Enhance") return;
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure?.apply(this, arguments);
            // The original node serialized fourteen widgets, ending with timeout_seconds.
            if (Array.isArray(info.widgets_values) && info.widgets_values.length === 14) {
                const sizing = this.widgets?.find((widget) => widget.name === "sizing_mode");
                if (sizing) sizing.value = "Multiplier";
            }
            return result;
        };
    },
});
