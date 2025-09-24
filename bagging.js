// Simple decision tree (for illustration, not optimal)
function trainSimpleTree(data, depth = 0, maxDepth = 4) {
    const classes = [...new Set(data.map(d => d[4]))];
    if (classes.length === 1 || depth >= maxDepth) {
        return {type: 'leaf', class: classes[0], count: data.length};
    }
    // Split on petal length (feature 2) at median
    const median = medianValue(data.map(d => d[2]));
    const left = data.filter(d => d[2] <= median);
    const right = data.filter(d => d[2] > median);
    return {
        type: 'node',
        feature: 2,
        threshold: median,
        left: trainSimpleTree(left, depth+1, maxDepth),
        right: trainSimpleTree(right, depth+1, maxDepth)
    };
}
function medianValue(arr) {
    const sorted = arr.slice().sort((a,b) => a-b);
    const mid = Math.floor(sorted.length/2);
    return sorted.length % 2 === 0 ? (sorted[mid-1]+sorted[mid])/2 : sorted[mid];
}

// Render tree with D3.js
function renderTree(tree, container) {
    container.innerHTML = '';
    const width = 700, height = 180;
    const svg = d3.create('svg').attr('width', width).attr('height', height);
    // Convert tree to D3 hierarchy
    function toD3(node) {
        if (node.type === 'leaf') {
            return {name: ''}; // No classification info
        }
        return {
            name: '', // No decision criteria
            children: [toD3(node.left), toD3(node.right)]
        };
    }
    const root = d3.hierarchy(toD3(tree));
    const treeLayout = d3.tree().size([width-40, height-40]);
    treeLayout(root);
    // Draw links
    svg.append('g')
        .selectAll('line')
        .data(root.links())
        .join('line')
        .attr('x1', d => d.source.x+20)
        .attr('y1', d => d.source.y+20)
        .attr('x2', d => d.target.x+20)
        .attr('y2', d => d.target.y+20)
        .attr('stroke', '#888');
    // Draw nodes
    svg.append('g')
        .selectAll('circle')
        .data(root.descendants())
        .join('circle')
        .attr('cx', d => d.x+20)
        .attr('cy', d => d.y+20)
        .attr('r', 16)
        .attr('fill', '#fff')
        .attr('stroke', '#333');
    // Draw labels
    svg.append('g')
        .selectAll('text')
        .data(root.descendants())
        .join('text')
        .attr('x', d => d.x+20)
        .attr('y', d => d.y+20)
        .attr('dy', 5)
        .attr('text-anchor', 'middle')
        .attr('font-size', '11px')
        .text(d => d.data.name);
    container.appendChild(svg.node());
}
// bagging.js
// Simple bagging: sample with replacement from T0 and train trees

function sampleWithReplacement(array, n) {
    const result = [];
    for (let i = 0; i < n; i++) {
        const idx = Math.floor(Math.random() * array.length);
        result.push(array[idx]);
    }
    return result;
}

function trainBaggedTrees(T0, numTrees = 5, treeDepth = 4) {
    const trees = [];
    for (let i = 0; i < numTrees; i++) {
        // Store indices of T0 used for this tree
        const sampleIndices = [];
        for (let j = 0; j < T0.length; j++) {
            const idx = Math.floor(Math.random() * T0.length);
            sampleIndices.push(idx);
        }
        const sample = sampleIndices.map(idx => T0[idx]);
        const tree = trainSimpleTree(sample, 0, treeDepth);
        trees.push({tree, sampleIndices});
    }
    return trees;
}

function predictTree(tree, sample) {
    let node = tree;
    while (node && node.type !== 'leaf') {
        const feat = node.feature ?? 2;
        if (Object.prototype.hasOwnProperty.call(node, 'threshold')) {
            const thr = node.threshold;
            node = sample[feat] <= thr ? node.left : node.right;
        } else if (Object.prototype.hasOwnProperty.call(node, 'value')) {
            const val = node.value;
            node = sample[feat] === val ? node.left : node.right;
        } else {
            break;
        }
    }
    return node?.class ?? 0;
}

// Train a tree on tuples dataset: rows like [p1,p2,...,pN,target]
function trainSimpleTreeOnTuples(data, depth = 0, maxDepth = 3) {
    const classes = [...new Set(data.map(d => d[d.length - 1]))];
    if (classes.length === 1 || depth >= maxDepth) {
        return { type: 'leaf', class: mostCommon(classes), count: data.length };
    }
    // Choose a feature among predictions (0..n-1). Use feature 0 median split for simplicity.
    const feature = 0;
    const median = medianValue(data.map(d => d[feature]));
    const left = data.filter(d => d[feature] <= median);
    const right = data.filter(d => d[feature] > median);
    if (left.length === 0 || right.length === 0) {
        return { type: 'leaf', class: mostCommon(data.map(d => d[d.length - 1])), count: data.length };
    }
    return {
        type: 'node',
        feature,
        threshold: median,
        left: trainSimpleTreeOnTuples(left, depth + 1, maxDepth),
        right: trainSimpleTreeOnTuples(right, depth + 1, maxDepth)
    };
}

function mostCommon(arr) {
    const counts = new Map();
    arr.forEach(v => counts.set(v, (counts.get(v) || 0) + 1));
    let best = null, bestC = -1;
    counts.forEach((c, v) => { if (c > bestC) { best = v; bestC = c; } });
    return best;
}

// Greedy tree on tuples: try all features (0..n-2) and all categorical values
function trainGreedyTreeOnTuples(data, depth = 0, maxDepth = 3) {
    const labels = data.map(d => d[d.length - 1]);
    const uniqueLabels = [...new Set(labels)];
    if (uniqueLabels.length === 1 || depth >= maxDepth) {
        return { type: 'leaf', class: mostCommon(labels), count: data.length };
    }
    const parentImp = giniImpurity(labels);
    const numFeatures = data[0].length - 1;
    let best = { gain: 0, feature: null, value: null, left: null, right: null };
    for (let f = 0; f < numFeatures; f++) {
        const vals = [...new Set(data.map(d => d[f]))];
        for (const v of vals) {
            const left = data.filter(d => d[f] === v);
            const right = data.filter(d => d[f] !== v);
            if (left.length === 0 || right.length === 0) continue;
            const imp = weightedImpurity(left, right);
            const gain = parentImp - imp;
            if (gain > best.gain) {
                best = { gain, feature: f, value: v, left, right };
            }
        }
    }
    if (best.gain <= 0 || best.feature === null) {
        return { type: 'leaf', class: mostCommon(labels), count: data.length };
    }
    return {
        type: 'node',
        feature: best.feature,
        // Use 'value' for equality split; renderer ignores text
        value: best.value,
        left: trainGreedyTreeOnTuples(best.left, depth + 1, maxDepth),
        right: trainGreedyTreeOnTuples(best.right, depth + 1, maxDepth)
    };
}

function giniImpurity(labels) {
    const n = labels.length;
    const counts = new Map();
    labels.forEach(l => counts.set(l, (counts.get(l) || 0) + 1));
    let sumSq = 0;
    counts.forEach(c => { const p = c / n; sumSq += p * p; });
    return 1 - sumSq;
}

function weightedImpurity(leftData, rightData) {
    const n = leftData.length + rightData.length;
    const leftImp = giniImpurity(leftData.map(d => d[d.length - 1]));
    const rightImp = giniImpurity(rightData.map(d => d[d.length - 1]));
    return (leftData.length / n) * leftImp + (rightData.length / n) * rightImp;
}
