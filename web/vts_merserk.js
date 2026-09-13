import { app } from "../../scripts/app.js";

const originalWidgets = [
    "server_url", "return_type", "upscaling_factor", "iterations", "nr_preset",
    "nr_style", "nr_intensity", "local_tone_strength", "local_structure_strength",
    "skin_structure_strength", "automatic_mask", "dlss_model_preset", "output_dir",
    "timeout_seconds",
];
const sizingWidgets = [
    ...originalWidgets, "enable_scaling", "enable_neural_rendering", "sizing_mode",
    "smallMaxSize", "largeMaxSize", "divisible_by", "crop", "scale_type",
    "dlss_quality", "vsr_quality",
];

app.registerExtension({
    name: "VTS.MerserkSizing",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VTS Merserk Enhance") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            this.vtsMerserkDefaults = new Map(this.widgets?.map(w => [w.name, w.value]));
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure?.apply(this, arguments);
            const values = info.widgets_values;
            // Older workflows stored widgets by position. Restore surviving controls
            // by name so removing presets cannot shift values into unrelated widgets.
            const names = !info.properties?.vts_merserk_schema && Array.isArray(values)
                ? (values.length === 14 ? originalWidgets : values.length === 24 ? sizingWidgets : null)
                : null;
            if (names) {
                const saved = new Map(names.map((name, index) => [name, values[index]]));
                if (names === originalWidgets) saved.set("sizing_mode", "Multiplier");
                for (const widget of this.widgets ?? []) {
                    if (saved.has(widget.name)) widget.value = saved.get(widget.name);
                    else if (this.vtsMerserkDefaults?.has(widget.name)) widget.value = this.vtsMerserkDefaults.get(widget.name);
                    if (widget.name === "upscaling_factor") widget.value = Number(widget.value);
                }
            }
            return result;
        };
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            const result = onSerialize?.apply(this, arguments);
            (info.properties ??= {}).vts_merserk_schema = 2;
            return result;
        };
    },
});
