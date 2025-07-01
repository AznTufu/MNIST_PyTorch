const canvas = document.getElementById("drawingCanvas");
const ctx = canvas.getContext("2d");
let drawing = false;
let session = null;

const labels = [
    "zéro", "un", "deux", "trois", "quatre",
    "cinq", "six", "sept", "huit", "neuf"
];

function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (e.touches ? e.touches[0].clientX : e.clientX) - rect.left,
        y: (e.touches ? e.touches[0].clientY : e.clientY) - rect.top,
    };
}

function startDraw(e) {
    drawing = true;
    const pos = getMousePos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
}

function draw(e) {
    if (!drawing) return;
    const pos = getMousePos(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.strokeStyle = "#000";
    ctx.lineWidth = 8;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.stroke();
}

function endDraw() {
    drawing = false;
    ctx.closePath();
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("mouseleave", endDraw);
canvas.addEventListener("touchstart", startDraw);
canvas.addEventListener("touchmove", draw);
canvas.addEventListener("touchend", endDraw);

document.getElementById("clearBtn").onclick = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById("predictedDigit").textContent = "?";
    document.getElementById("confidence").textContent = "Dessinez un chiffre et cliquez sur Prédire !";

    const bars = document.querySelectorAll('.prob-bar');
    bars.forEach((bar, index) => {
        const fill = bar.querySelector('.prob-fill');
        const value = bar.querySelector('.prob-value');
        if (fill) fill.style.width = '0%';
        if (value) value.textContent = '0%';
    });

    const resultDiv = document.querySelector('.prediction-result');
    if (resultDiv) {
        resultDiv.classList.remove('correct', 'confident', 'uncertain');
    }
};

function preprocessCanvas() {
    const src = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = src.data;

    let minX = canvas.width,
        minY = canvas.height,
        maxX = 0,
        maxY = 0;
    let found = false;
    
    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const i = (y * canvas.width + x) * 4;
            const r = data[i],
                g = data[i + 1],
                b = data[i + 2];
            const v = (r + g + b) / 3;
            if (v < 250) {
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                found = true;
            }
        }
    }

    if (!found) return new Float32Array(28 * 28).fill(-1);

    const boxW = maxX - minX + 1;
    const boxH = maxY - minY + 1;
    const temp = document.createElement("canvas");
    temp.width = 28;
    temp.height = 28;
    const tctx = temp.getContext("2d");
    tctx.fillStyle = "white";
    tctx.fillRect(0, 0, 28, 28);
    
    const scale = Math.min(20 / boxW, 20 / boxH);
    const w = boxW * scale;
    const h = boxH * scale;
    const dx = (28 - w) / 2;
    const dy = (28 - h) / 2;
    tctx.drawImage(canvas, minX, minY, boxW, boxH, dx, dy, w, h);

    const imgData = tctx.getImageData(0, 0, 28, 28).data;
    const input = [];
    for (let i = 0; i < imgData.length; i += 4) {
        let v = (imgData[i] + imgData[i + 1] + imgData[i + 2]) / 3;
        v = 255 - v;
        v = (v / 255 - 0.5) / 0.5;
        input.push(v);
    }
    return new Float32Array(input);
}

async function loadModel() {
    if (!session) {
        console.log("Chargement du modèle ONNX...");
        session = await ort.InferenceSession.create("cnn_model.onnx");
        console.log("Modèle chargé !");
    }
    return session;
}

function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const expValues = logits.map(x => Math.exp(x - maxLogit));
    const sumExp = expValues.reduce((a, b) => a + b, 0);
    return expValues.map(x => x / sumExp);
}

function updateProbabilityBars(probabilities) {
    const bars = document.querySelectorAll('.prob-bar');
    
    bars.forEach((bar, index) => {
        const prob = probabilities[index] || 0;
        const percentage = (prob * 100).toFixed(1);
        
        const fill = bar.querySelector('.prob-fill');
        const value = bar.querySelector('.prob-value');
        
        if (fill) fill.style.width = `${percentage}%`;
        if (value) value.textContent = `${percentage}%`;
    });
}

async function predict() {
    if (typeof ort === 'undefined') {
        document.getElementById("confidence").textContent = "ONNX Runtime non disponible !";
        return;
    }
    
    document.getElementById("confidence").textContent = "Prédiction...";
    document.getElementById("predictBtn").disabled = true;
    document.getElementById("predictBtn").textContent = "Prédiction...";
    
    const inputTensor = preprocessCanvas();
    const tensor = new ort.Tensor("float32", inputTensor, [1, 1, 28, 28]);
    
    try {
        const sess = await loadModel();
        const feeds = { input: tensor };
        const results = await sess.run(feeds);
        const output = results.output.data;
        
        // Argmax
        let maxIdx = 0,
            maxVal = output[0];
        for (let i = 1; i < output.length; ++i) {
            if (output[i] > maxVal) {
                maxVal = output[i];
                maxIdx = i;
            }
        }

        const probabilities = softmax(Array.from(output));
        const confidence = probabilities[maxIdx];

        document.getElementById("predictedDigit").textContent = maxIdx;
        const confidenceText = `${labels[maxIdx]} (${(confidence * 100).toFixed(1)}%)`;
        document.getElementById("confidence").textContent = confidenceText;

        updateProbabilityBars(probabilities);

        const resultDiv = document.querySelector('.prediction-result');
        if (resultDiv) {
            resultDiv.classList.remove('correct', 'confident', 'uncertain');
            if (confidence > 0.8) {
                resultDiv.classList.add('confident');
            } else if (confidence < 0.5) {
                resultDiv.classList.add('uncertain');
            }
        }
        
        console.log(`Prédiction: ${maxIdx} (${labels[maxIdx]}) avec ${(confidence * 100).toFixed(1)}% de confiance`);
        
    } catch (e) {
        document.getElementById("confidence").textContent = "Erreur: " + e;
        console.error("Erreur de prédiction:", e);
    } finally {
        document.getElementById("predictBtn").disabled = false;
        document.getElementById("predictBtn").textContent = "Prédire";
    }
}

document.getElementById("predictBtn").onclick = predict;

function initProbabilityBars() {
    const container = document.getElementById('probBars');
    if (!container) return;
    
    container.innerHTML = '';
    
    for (let i = 0; i < 10; i++) {
        const barDiv = document.createElement('div');
        barDiv.className = 'prob-bar';
        barDiv.innerHTML = `
            <div class="prob-label">${i}</div>
            <div class="prob-fill-container">
                <div class="prob-fill" style="width: 0%"></div>
            </div>
            <div class="prob-value">0%</div>
        `;
        container.appendChild(barDiv);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('Application de reconnaissance de chiffres initialisée');
    console.log('ONNX Runtime disponible:', typeof ort !== 'undefined');

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    initProbabilityBars();
    
    console.log('Prêt à dessiner !');
});
