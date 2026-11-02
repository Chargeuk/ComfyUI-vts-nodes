const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");
let extension;
const source = fs.readFileSync(path.join(__dirname, "../web/vts_colour_context.js"), "utf8")
    .replace(/^import .*;\r?\n/, "");
vm.runInNewContext(source, { app: { registerExtension(value) { extension = value; } } });

(async () => {
    // Representative ordered widgets from the decoder; sockets have no values.
    const defaults = {
        tile_size_x:512, tile_size_y:512, overlap:64, temporal_size:64, temporal_overlap:8,
        return_type:"Tensor", prefix:"vae_decode", start_sequence:0, output_dir:"./tmp", format:"png",
        num_workers:16, compression_level:9, quality:95,
        fast_vae_name:"None", fast_vae_keep_loaded:false, color_match_method:"reinhard_lab_gpu",
        color_match_weight:0.5, white_balance_weight:0, brightness_method:"gamma", brightness_weight:0,
        contrast_weight:0, calculation_mode:"fixed_per_clip", smoothing:0.9, overall_weight:1,
        analysis_size:128, lut_resolution:33, context_source:"None", context_length:"22", enable_merserk:false,
        merserk_server_url:"http://default", merserk_enable_frame_interpolation:false,
        merserk_interpolation_multiplier:2, context_return_type:"Tensor", context_output_dir:"./tmp/disklatents",
        context_prefix:"context", context_start_sequence:0, context_compression_level:3,
        latent_device_policy:"Original", latent_device:"cpu",
    };
    const oldNames = [
        "tile_size_x", "tile_size_y", "overlap", "temporal_size", "temporal_overlap",
        "return_type", "prefix", "start_sequence", "output_dir", "format", "num_workers", "compression_level", "quality",
        "fast_vae_name", "fast_vae_keep_loaded", "color_match_method", "color_match_weight", "white_balance_weight",
        "brightness_method", "brightness_weight", "contrast_weight", "calculation_mode", "smoothing", "overall_weight",
        "analysis_size", "lut_resolution", "encode_corrected_context", "context_length", "use_merserk_for_context",
        "context_frame_selection", "enable_merserk", "merserk_server_url", "merserk_enable_frame_interpolation",
        "merserk_interpolation_multiplier", "context_return_type", "context_output_dir", "context_prefix",
        "context_start_sequence", "context_compression_level", "latent_device_policy", "latent_device",
    ];
    function Node() {
        this.widgets = Object.entries(defaults).map(([name, value]) => ({name, value}));
        this.onNodeCreated();
    }
    let configured=0, serialized=0;
    Node.prototype.onConfigure=function() { configured++; };
    Node.prototype.onSerialize=function() { serialized++; };
    await extension.beforeRegisterNodeDef(Node, {name:"VTS VAE Decode Tiled Colour Match"});
    for (const fast of [false, true]) {
        for (const storage of [false, true]) {
            for (const [encode, merserk, use, selection, interpolate, expected] of [
                [false, true, true, "Interpolated", true, "None"],
                [true, false, true, "Interpolated", true, "Colour-corrected frames"],
                [true, true, false, "Interpolated", true, "Colour-corrected frames"],
                [true, true, true, "Original", true, "Merserk — original frames"],
                [true, true, true, "Interpolated", false, "Merserk — original frames"],
                [true, true, true, "Interpolated", true, "Merserk — interpolated frames"],
            ]) {
                const names = oldNames.filter(name => (fast || !name.startsWith("fast_vae_"))
                    && (storage || !["context_return_type", "context_output_dir", "context_prefix",
                        "context_start_sequence", "context_compression_level", "latent_device_policy", "latent_device"].includes(name)));
                const saved = {...defaults, tile_size_x:768, temporal_overlap:0, return_type:"DiskImage", prefix:"saved image",
                    start_sequence:31, fast_vae_name:"vae_fast/test.safetensors", fast_vae_keep_loaded:true,
                    color_match_weight:0.7, encode_corrected_context:encode, context_length:"39", use_merserk_for_context:use,
                    context_frame_selection:selection, enable_merserk:merserk, merserk_enable_frame_interpolation:interpolate,
                    merserk_server_url:"http://saved", merserk_interpolation_multiplier:4, context_return_type:"DiskLatent",
                    context_output_dir:"./saved", context_prefix:"saved context", context_start_sequence:12,
                    context_compression_level:7, latent_device_policy:"CPU"};
                const values = names.map(name => saved[name]);
                const node = new Node();
                node.widgets.forEach((w, i) => { w.value = values[i]; }); // LiteGraph's initial positional assignment.
                node.onConfigure({widgets_values:values});
                const actual = Object.fromEntries(node.widgets.map(w => [w.name, w.value]));
                assert.equal(actual.context_source, expected);
                for (const name of Object.keys(defaults).filter(name => name !== "context_source")) {
                    assert.equal(actual[name], names.includes(name) ? saved[name] : defaults[name], name);
                }
                const info = {}; node.onSerialize(info);
                assert.equal(info.properties.vts_colour_context_schema, 1);
            }
        }
    }
    // Current dropdown values survive both marked and unmarked new workflows.
    for (const marked of [false, true]) {
        const node = new Node();
        const current = {...defaults, context_source:"Provided context samples", context_length:"56"};
        node.widgets.forEach(w => { w.value = current[w.name]; });
        node.onConfigure({widgets_values:Object.values(current), properties: marked ? {vts_colour_context_schema:1} : {}});
        assert.equal(node.widgets.find(w => w.name === "context_source").value, "Provided context samples");
        assert.equal(node.widgets.find(w => w.name === "context_length").value, "56");
    }
    assert.equal(configured,26); assert.equal(serialized,24);
    console.log("PASS 24 legacy widget migrations and 2 current workflow layouts");
})();
