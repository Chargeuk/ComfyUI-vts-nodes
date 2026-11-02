import { app } from "../../scripts/app.js";

const sources = ["None", "Provided context samples", "Decoded frames (before corrections+Merserk)",
    "Colour-corrected frames", "Merserk — original frames", "Merserk — interpolated frames"];

app.registerExtension({
    name: "VTS.ColourContextSource",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VTS VAE Decode Tiled Colour Match") return;
        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onCreated?.apply(this, arguments);
            this.vtsColourContextDefaults = new Map(this.widgets?.map(w => [w.name, w.value]));
            return result;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const result = onConfigure?.apply(this, arguments);
            const values = info.widgets_values;
            if (info.properties?.vts_colour_context_schema || !Array.isArray(values)) return result;
            const current = [...(this.vtsColourContextDefaults?.keys() ?? [])];
            if (sources.includes(values[current.indexOf("context_source")])) return result;
            const old = current.filter(name => name !== "context_source");
            old.splice(old.indexOf("context_length"), 0, "encode_corrected_context");
            old.splice(old.indexOf("context_length") + 1, 0, "use_merserk_for_context", "context_frame_selection");
            // Fast decode and disk storage were optional additions. Restore old
            // positional arrays by their original names before assigning new widgets.
            const candidates = [];
            for (const fast of [true, false]) {
                for (const storage of [true, false]) {
                    candidates.push(old.filter(name => (fast || !name.startsWith("fast_vae_"))
                        && (storage || !(name.startsWith("latent_") || (name.startsWith("context_")
                            && !["context_length", "context_frame_selection"].includes(name))))));
                }
            }
            const names = candidates.find(names => names.length === values.length
                && typeof values[names.indexOf("encode_corrected_context")] === "boolean");
            if (!names) return result;
            const saved = new Map(names.map((name, index) => [name, values[index]]));
            let source = "None";
            if (saved.get("encode_corrected_context")) {
                source = "Colour-corrected frames";
                if (saved.get("enable_merserk") && saved.get("use_merserk_for_context")) {
                    source = saved.get("context_frame_selection") === "Interpolated"
                        && saved.get("merserk_enable_frame_interpolation")
                        ? "Merserk — interpolated frames" : "Merserk — original frames";
                }
            }
            saved.set("context_source", source);
            for (const widget of this.widgets ?? []) {
                widget.value = saved.has(widget.name) ? saved.get(widget.name)
                    : this.vtsColourContextDefaults.get(widget.name);
            }
            return result;
        };
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (info) {
            const result = onSerialize?.apply(this, arguments);
            (info.properties ??= {}).vts_colour_context_schema = 1;
            return result;
        };
    },
});
