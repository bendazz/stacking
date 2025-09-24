let T0;
let trees;
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
        const targetCircle = document.createElement('span');
        targetCircle.className = `circle class-${target}`;
        targetCircle.title = `True class: ${target}`;
        targetCircle.style.display = 'inline-block';
        targetCircle.style.marginLeft = '12px';
        tupleContainer.appendChild(targetCircle);
        tuplesPanel.appendChild(tupleContainer);
    });
}

function renderT1(T1) {
    const setDiv = document.getElementById('setT1');
    if (!setDiv) return;
    setDiv.innerHTML = '';
    T1.forEach(sample => {
        const circle = document.createElement('span');
        circle.className = `circle class-${sample[4]}`;
        circle.title = `Class: ${sample[4]}`;
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
    renderAllTuples();
})();
