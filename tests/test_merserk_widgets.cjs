const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const assert = require("node:assert/strict");
let extension;
const source = fs.readFileSync(path.join(__dirname, "../web/vts_merserk.js"), "utf8").replace(/^import .*;\r?\n/, "");
vm.runInNewContext(source, { app: { registerExtension(value) { extension = value; } } });
(async () => {
    function Node() {
        this.widgets = Object.entries({server_url:"http://default", iterations:1, nr_passes:1,
            nr_style:"Default", nr_color_strength:1, tone_preservation:0,
            upscaling_factor:1, sizing_mode:"Scale to Min", vsr_quality:"Ultra"})
            .map(([name,value]) => ({name,value}));
        this.onNodeCreated();
    }
    let configured=0, serialized=0;
    Node.prototype.onConfigure=function() { configured++; };
    Node.prototype.onSerialize=function() { serialized++; };
    await extension.beforeRegisterNodeDef(Node,{name:"VTS Merserk Enhance"});
    const old=["http://saved","Tensor","2",3,"Preset #2","Cinematic",1.3,1.2,1.1,-1,false,"K","",900];
    const sized=[...old,true,true,"Scale to Min",720,1280,2,"center","small","Performance","High"];
    for(const values of [old,sized]) {
        const node=new Node();
        node.widgets.forEach((w,i) => {w.value=values[i];});
        node.onConfigure({widgets_values:values});
        const actual=Object.fromEntries(node.widgets.map(w=>[w.name,w.value]));
        assert.equal(actual.server_url,"http://saved");
        assert.equal(actual.iterations,3);
        assert.equal(actual.nr_passes,1);
        assert.equal(actual.nr_style,"Cinematic");
        assert.equal(actual.nr_color_strength,1);
        assert.equal(actual.tone_preservation,0);
        assert.equal(actual.upscaling_factor,2);
        assert.equal(actual.sizing_mode,values.length===14?"Multiplier":"Scale to Min");
        assert.equal(actual.vsr_quality,values.length===14?"Ultra":"High");
        const saved={}; node.onSerialize(saved);
        assert.equal(saved.properties.vts_merserk_schema,2);
    }
    const current=new Node();
    current.widgets.find(w=>w.name==="nr_passes").value=4;
    current.onConfigure({widgets_values:sized,properties:{vts_merserk_schema:2}});
    assert.equal(current.widgets.find(w=>w.name==="nr_passes").value,4);
    assert.equal(configured,3); assert.equal(serialized,2);
    console.log("PASS old widget layouts restore surviving values and new loop defaults without notices");
})();
