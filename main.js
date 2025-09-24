let T0;
let trees;
let stackedTree;
// Utility: Fisher-Yates shuffle
function shuffle(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}
function renderAllTuples() {
    const tuplesPanel = document.getElementById('tuplesPanel');
    if (!tuplesPanel) return;
    tuplesPanel.innerHTML = '';
    if (!window.T1_for_tuples || !trees || typeof predictTree !== 'function') return;
    const tuples = window.T1_for_tuples.map(sample => {
        const preds = trees.map(obj => predictTree(obj.tree, sample));
        return { preds, target: sample[4] };
    });
    tuples.forEach(({ preds, target }) => {
        const tupleContainer = document.createElement('div');
        tupleContainer.style.display = 'flex';
        tupleContainer.style.alignItems = 'center';
        tupleContainer.style.gap = '8px';
        tupleContainer.style.border = '2px solid #bbb';
        tupleContainer.style.borderRadius = '24px';
        tupleContainer.style.padding = '8px 18px';
        tupleContainer.style.background = '#f9f9f9';
        preds.forEach(pred => {
            const circle = document.createElement('span');
            circle.className = `circle class-${pred}`;
            circle.title = `Predicted class: ${pred}`;
            circle.style.display = 'inline-block';
            tupleContainer.appendChild(circle);
        });
        const arrow = document.createElement('span');
        arrow.textContent = '→';
        arrow.style.fontSize = '18px';
        arrow.style.lineHeight = '18px';
        arrow.style.color = '#333';
        arrow.style.margin = '0 6px';
        arrow.style.display = 'inline-block';
        tupleContainer.appendChild(arrow);
        const targetCircle = document.createElement('span');
        targetCircle.className = `circle class-${target}`;
        targetCircle.title = `True class (target): ${target}`;
        targetCircle.style.display = 'inline-block';
        targetCircle.style.marginLeft = '4px';
        targetCircle.style.border = '3px solid #000';
        targetCircle.style.boxShadow = '0 0 6px rgba(0,0,0,0.45)';
        tupleContainer.appendChild(targetCircle);
        tuplesPanel.appendChild(tupleContainer);
    });
}

function buildTuplesDataset() {
    if (!window.T1_for_tuples || !trees || typeof predictTree !== 'function') return [];
    return window.T1_for_tuples.map(sample => {
        const preds = trees.map(obj => predictTree(obj.tree, sample));
        return [...preds, sample[4]]; // last element is target
    });
}

function renderStackedTreeOnTuples() {
    const panel = document.getElementById('stackedTreePanel');
    if (!panel) return;
    panel.innerHTML = '';
    const dataset = buildTuplesDataset();
    if (!dataset.length) return;
    // Train the greedy tuples tree to purity (no fixed depth limit)
    stackedTree = trainGreedyTreeOnTuples(dataset, 0, Infinity);
    const container = document.createElement('div');
    renderTree(stackedTree, container);
    panel.appendChild(container);
}

function renderT1(T1) {
    const setDiv = document.getElementById('setT1');
    if (!setDiv) return;
    setDiv.innerHTML = '';
    // Ensure the trees are visible under panel 2
    renderBaggedTreesForT1(trees || []);
    T1.forEach((sample, idx) => {
        const circle = document.createElement('span');
        circle.className = `circle class-${sample[4]}`;
        circle.title = `Class: ${sample[4]} (click to show predictions)`;
        circle.style.cursor = 'pointer';
        circle.onclick = () => {
            clearT1Predictions();
            updateT1Predictions(sample);
            renderTupleForSample(sample);
        };
        setDiv.appendChild(circle);
    });
}

function trainTestSplit(data, testRatio = 0.2) {
    const indices = Array.from(data.keys());
    shuffle(indices);
    const testSize = Math.floor(data.length * testRatio);
    const testIdx = indices.slice(0, testSize);
    const trainIdx = indices.slice(testSize);
    return {
        train: trainIdx.map(i => data[i]),
        test: testIdx.map(i => data[i]),
        trainIndices: trainIdx,
        testIndices: testIdx
    };}

function splitTrainSet(train, trainIndices) {
    const mid = Math.floor(train.length / 2);
    T0 = train.slice(0, mid);
    const T0Indices = trainIndices.slice(0, mid);
    const T1 = train.slice(mid);
    const T1Indices = trainIndices.slice(mid);
    return { T0, T0Indices, T1, T1Indices };}

function renderT0(T0) {
    const setDiv = document.getElementById('setT0');
    setDiv.innerHTML = '';
    const highlightIndices = window.highlightT0Indices || [];
    T0.forEach((sample, i) => {
        const circle = document.createElement('span');
        circle.className = `circle class-${sample[4]}`;
        circle.title = `Class: ${sample[4]}`;
        if (highlightIndices.includes(i)) {
            circle.style.border = '2px solid #ff9800';
            circle.style.boxShadow = '0 0 8px #ff9800';
        }
        setDiv.appendChild(circle);
    });}

function renderBaggedTrees(T0, trees) {
    // Remove previous tree display if any
    let treesDiv = document.getElementById('baggedTrees');
    if (!treesDiv) {
        treesDiv = document.createElement('div');
        treesDiv.id = 'baggedTrees';
        treesDiv.style.display = 'flex';
        treesDiv.style.justifyContent = 'center';
        treesDiv.style.gap = '32px';
        treesDiv.style.margin = '32px 0';
        const panel1 = document.getElementById('baggedTreesPanel');
        if (panel1) {
            panel1.appendChild(treesDiv);
        } else {
            document.body.appendChild(treesDiv);
        }
    }
    treesDiv.innerHTML = '';
    trees.forEach((obj, i) => {
        const treeContainer = document.createElement('div');
        treeContainer.style.background = '#fff';
        treeContainer.style.borderRadius = '12px';
        treeContainer.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)';
        treeContainer.style.padding = '12px';
        treeContainer.style.minWidth = '220px';
        treeContainer.style.display = 'flex';
        treeContainer.style.flexDirection = 'column';
        treeContainer.style.alignItems = 'center';
        const title = document.createElement('div');
        title.textContent = `Tree ${i+1}`;
        title.style.textAlign = 'center';
        title.style.fontWeight = 'bold';
        title.style.marginBottom = '8px';
        treeContainer.appendChild(title);
        const svgDiv = document.createElement('div');
        renderTree(obj.tree, svgDiv);
        treeContainer.appendChild(svgDiv);
        treeContainer.style.cursor = 'pointer';
        treeContainer.onclick = function(e) {
            window.highlightT0Indices = obj.sampleIndices;
            renderT0(T0);
        };
        treesDiv.appendChild(treeContainer);
    });
}

function renderBaggedTreesForT1(trees) {
    const panel = document.getElementById('baggedTreesPanel2');
    if (!panel) return;
    panel.innerHTML = '';
    const treesDiv = document.createElement('div');
    treesDiv.style.display = 'flex';
    treesDiv.style.justifyContent = 'center';
    treesDiv.style.gap = '32px';
    treesDiv.style.flexWrap = 'wrap';
    panel.appendChild(treesDiv);
    trees.forEach((obj, i) => {
        const group = document.createElement('div');
        group.style.display = 'flex';
        group.style.flexDirection = 'column';
        group.style.alignItems = 'center';

        const col = document.createElement('div');
        col.style.display = 'flex';
        col.style.flexDirection = 'column';
        col.style.alignItems = 'center';
        col.style.background = '#fff';
        col.style.borderRadius = '12px';
        col.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)';
        col.style.padding = '12px';
        col.style.minWidth = '220px';

        const title = document.createElement('div');
        title.textContent = `Tree ${i+1}`;
        title.style.textAlign = 'center';
        title.style.fontWeight = 'bold';
        title.style.marginBottom = '8px';
        col.appendChild(title);

        const svgDiv = document.createElement('div');
        renderTree(obj.tree, svgDiv);
        col.appendChild(svgDiv);

    const predHolder = document.createElement('div');
    predHolder.id = `t1-pred-holder-${i}`;
    predHolder.style.marginTop = '8px';
    predHolder.style.minHeight = '40px';
    predHolder.style.display = 'flex';
    predHolder.style.flexDirection = 'column';
    predHolder.style.alignItems = 'center';
    predHolder.style.justifyContent = 'flex-start';

        group.appendChild(col);
        group.appendChild(predHolder);
        treesDiv.appendChild(group);
    });
}

function clearT1Predictions() {
    if (!trees) return;
    for (let i = 0; i < trees.length; i++) {
        const holder = document.getElementById(`t1-pred-holder-${i}`);
        if (holder) holder.innerHTML = '';
    }
}

function updateT1Predictions(sample) {
    if (!trees || typeof predictTree !== 'function') return;
    for (let i = 0; i < trees.length; i++) {
        const holder = document.getElementById(`t1-pred-holder-${i}`);
        if (!holder) continue;
        holder.innerHTML = '';
    const pred = predictTree(trees[i].tree, sample);
    const circle = document.createElement('span');
    circle.className = `circle class-${pred}`;
    circle.title = `Predicted class: ${pred}`;
    holder.appendChild(circle);
    const arrow = document.createElement('div');
    arrow.textContent = '↓';
    arrow.style.fontSize = '16px';
    arrow.style.lineHeight = '16px';
    arrow.style.color = '#666';
    arrow.style.marginTop = '4px';
    holder.appendChild(arrow);
    }
}

function renderTupleForSample(sample) {
    const tuplesPanel = document.getElementById('t1TuplePanel');
    if (!tuplesPanel) return;
    tuplesPanel.innerHTML = '';
    if (!trees || typeof predictTree !== 'function') return;
    const preds = trees.map(obj => predictTree(obj.tree, sample));
    const tupleContainer = document.createElement('div');
    tupleContainer.style.display = 'flex';
    tupleContainer.style.alignItems = 'center';
    tupleContainer.style.gap = '8px';
    tupleContainer.style.border = '2px solid #bbb';
    tupleContainer.style.borderRadius = '24px';
    tupleContainer.style.padding = '8px 18px';
    tupleContainer.style.background = '#f9f9f9';
    preds.forEach(pred => {
        const c = document.createElement('span');
        c.className = `circle class-${pred}`;
        c.title = `Predicted class: ${pred}`;
        c.style.display = 'inline-block';
        tupleContainer.appendChild(c);
    });
    // Arrow pointing to the target
    const arrow = document.createElement('span');
    arrow.textContent = '→';
    arrow.style.fontSize = '18px';
    arrow.style.lineHeight = '18px';
    arrow.style.color = '#333';
    arrow.style.margin = '0 6px';
    arrow.style.display = 'inline-block';
    tupleContainer.appendChild(arrow);
    const target = sample[4];
    const targetCircle = document.createElement('span');
    targetCircle.className = `circle class-${target}`;
    targetCircle.title = `True class (target): ${target}`;
    targetCircle.style.display = 'inline-block';
    targetCircle.style.marginLeft = '4px';
    targetCircle.style.border = '3px solid #000';
    targetCircle.style.boxShadow = '0 0 6px rgba(0,0,0,0.45)';
    tupleContainer.appendChild(targetCircle);
    tuplesPanel.appendChild(tupleContainer);
}

// Panel 4: Test set (T2)
function renderBaggedTreesForT2(trees) {
    const panel = document.getElementById('baggedTreesPanel3');
    if (!panel) return;
    panel.innerHTML = '';
    const treesDiv = document.createElement('div');
    treesDiv.style.display = 'flex';
    treesDiv.style.justifyContent = 'center';
    treesDiv.style.gap = '32px';
    treesDiv.style.flexWrap = 'wrap';
    panel.appendChild(treesDiv);
    trees.forEach((obj, i) => {
        const group = document.createElement('div');
        group.style.display = 'flex';
        group.style.flexDirection = 'column';
        group.style.alignItems = 'center';
        const col = document.createElement('div');
        col.style.display = 'flex';
        col.style.flexDirection = 'column';
        col.style.alignItems = 'center';
        col.style.background = '#fff';
        col.style.borderRadius = '12px';
        col.style.boxShadow = '0 2px 8px rgba(0,0,0,0.08)';
        col.style.padding = '12px';
        col.style.minWidth = '220px';
        const title = document.createElement('div');
        title.textContent = `Tree ${i+1}`;
        title.style.textAlign = 'center';
        title.style.fontWeight = 'bold';
        title.style.marginBottom = '8px';
        col.appendChild(title);
        const svgDiv = document.createElement('div');
        renderTree(obj.tree, svgDiv);
        col.appendChild(svgDiv);
        const predHolder = document.createElement('div');
        predHolder.id = `t2-pred-holder-${i}`;
        predHolder.style.marginTop = '8px';
        predHolder.style.minHeight = '40px';
        predHolder.style.display = 'flex';
        predHolder.style.flexDirection = 'column';
        predHolder.style.alignItems = 'center';
        predHolder.style.justifyContent = 'flex-start';
        group.appendChild(col);
        group.appendChild(predHolder);
        treesDiv.appendChild(group);
    });
}

function clearT2Predictions() {
    if (!trees) return;
    for (let i = 0; i < trees.length; i++) {
        const holder = document.getElementById(`t2-pred-holder-${i}`);
        if (holder) holder.innerHTML = '';
    }
}

function updateT2Predictions(sample) {
    if (!trees || typeof predictTree !== 'function') return;
    for (let i = 0; i < trees.length; i++) {
        const holder = document.getElementById(`t2-pred-holder-${i}`);
        if (!holder) continue;
        holder.innerHTML = '';
        const pred = predictTree(trees[i].tree, sample);
        const circle = document.createElement('span');
        circle.className = `circle class-${pred}`;
        circle.title = `Predicted class: ${pred}`;
        holder.appendChild(circle);
        const arrow = document.createElement('div');
        arrow.textContent = '↓';
        arrow.style.fontSize = '16px';
        arrow.style.lineHeight = '16px';
        arrow.style.color = '#666';
        arrow.style.marginTop = '4px';
        holder.appendChild(arrow);
    }
}

function renderT2TupleForSample(sample) {
    const t2Panel = document.getElementById('t2TuplePanel');
    if (!t2Panel) return;
    t2Panel.innerHTML = '';
    if (!trees || typeof predictTree !== 'function') return;
    const preds = trees.map(obj => predictTree(obj.tree, sample));
    const tupleContainer = document.createElement('div');
    tupleContainer.style.display = 'flex';
    tupleContainer.style.alignItems = 'center';
    tupleContainer.style.gap = '8px';
    tupleContainer.style.border = '2px solid #bbb';
    tupleContainer.style.borderRadius = '24px';
    tupleContainer.style.padding = '8px 18px';
    tupleContainer.style.background = '#f9f9f9';
    preds.forEach(pred => {
        const c = document.createElement('span');
        c.className = `circle class-${pred}`;
        c.title = `Predicted class: ${pred}`;
        c.style.display = 'inline-block';
        tupleContainer.appendChild(c);
    });
    const arrow = document.createElement('span');
    arrow.textContent = '→';
    arrow.style.fontSize = '18px';
    arrow.style.lineHeight = '18px';
    arrow.style.color = '#333';
    arrow.style.margin = '0 6px';
    arrow.style.display = 'inline-block';
    tupleContainer.appendChild(arrow);
    const target = sample[4];
    const targetCircle = document.createElement('span');
    targetCircle.className = `circle class-${target}`;
    targetCircle.title = `True class (target): ${target}`;
    targetCircle.style.display = 'inline-block';
    targetCircle.style.marginLeft = '4px';
    targetCircle.style.border = '3px solid #000';
    targetCircle.style.boxShadow = '0 0 6px rgba(0,0,0,0.45)';
    tupleContainer.appendChild(targetCircle);
    t2Panel.appendChild(tupleContainer);
}

function renderT2MetaForSample(sample) {
    const metaPanel = document.getElementById('t2MetaPanel');
    if (!metaPanel) return;
    metaPanel.innerHTML = '';
    // Downward arrow under tuple
    const down = document.createElement('div');
    down.textContent = '↓';
    down.style.fontSize = '18px';
    down.style.color = '#666';
    metaPanel.appendChild(down);
    // Render the meta-learner trained on Panel 3 tuples
    const treeWrap = document.createElement('div');
    renderTree(stackedTree, treeWrap);
    metaPanel.appendChild(treeWrap);
    // Prediction under the meta tree
    const preds = trees.map(obj => predictTree(obj.tree, sample));
    const metaPred = predictTree(stackedTree, preds);
    const predHolder = document.createElement('div');
    predHolder.style.marginTop = '8px';
    predHolder.style.display = 'flex';
    predHolder.style.justifyContent = 'center';
    const circle = document.createElement('span');
    circle.className = `circle class-${metaPred}`;
    circle.title = `Meta prediction: ${metaPred}`;
    predHolder.appendChild(circle);
    metaPanel.appendChild(predHolder);
    // Update overall test accuracy display as well
    renderT2Accuracy(window.T2_test || []);
}

function renderT2Accuracy(test) {
    const accDiv = document.getElementById('t2Accuracy');
    if (!accDiv || !stackedTree || !trees || !test.length) return;
    let correct = 0;
    for (const s of test) {
        const preds = trees.map(obj => predictTree(obj.tree, s));
        const metaPred = predictTree(stackedTree, preds);
        if (metaPred === s[4]) correct++;
    }
    const acc = (correct / test.length) * 100;
    accDiv.textContent = `Meta-learner accuracy on test: ${acc.toFixed(1)}% (${correct}/${test.length})`;
}

function renderT2BaseAccuracies(test) {
    const container = document.getElementById('t2BaseAccuracies');
    if (!container || !trees || !test.length) return;
    container.innerHTML = '';
    for (let i = 0; i < trees.length; i++) {
        let correct = 0;
        for (const s of test) {
            const pred = predictTree(trees[i].tree, s);
            if (pred === s[4]) correct++;
        }
        const acc = (correct / test.length) * 100;
        const card = document.createElement('div');
        card.style.background = '#fff';
        card.style.border = '1px solid #ddd';
        card.style.borderRadius = '8px';
        card.style.padding = '8px 12px';
        card.style.minWidth = '160px';
        card.style.textAlign = 'center';
        card.style.boxShadow = '0 1px 4px rgba(0,0,0,0.06)';
        const title = document.createElement('div');
        title.textContent = `Tree ${i+1}`;
        title.style.fontWeight = '600';
        title.style.marginBottom = '4px';
        const val = document.createElement('div');
        val.textContent = `${acc.toFixed(1)}% (${correct}/${test.length})`;
        card.appendChild(title);
        card.appendChild(val);
        container.appendChild(card);
    }
}

function renderT2(test) {
    const setDiv = document.getElementById('setT2');
    if (!setDiv) return;
    setDiv.innerHTML = '';
    renderBaggedTreesForT2(trees || []);
    test.forEach(sample => {
        const circle = document.createElement('span');
        circle.className = `circle class-${sample[4]}`;
        circle.title = `Class: ${sample[4]} (click to show predictions)`;
        circle.style.cursor = 'pointer';
        circle.onclick = () => {
            clearT2Predictions();
            updateT2Predictions(sample);
            renderT2TupleForSample(sample);
            renderT2MetaForSample(sample);
        };
        setDiv.appendChild(circle);
    });
}

(function() {
    // Use IRIS_DATA from iris.js
    const split = trainTestSplit(IRIS_DATA, 0.2); // 80/20 split
    const train = split.train;
    const trainIndices = split.trainIndices;
    const test = split.test;
    window.T1_for_tuples = null;
    const splitResult = splitTrainSet(train, trainIndices);
    T0 = splitResult.T0;
    const T1 = splitResult.T1;
    window.T1_for_tuples = T1;
    window.T2_test = test;
    // Train bagged trees now that T0 is defined
    trees = trainBaggedTrees(T0, 5, 1);
    renderT0(T0);
    renderT1(T1);
    renderBaggedTrees(T0, trees);
    renderBaggedTreesForT1(trees);
    renderAllTuples();
    renderStackedTreeOnTuples();
    renderT2(test);
    renderT2Accuracy(test);
    renderT2BaseAccuracies(test);
})();
