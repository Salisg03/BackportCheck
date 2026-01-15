// CONFIGURATION 
const BACKEND_URL = "http://localhost:5000/predict";
let currentThreshold = 0.50; 

// Écoute les changements depuis le popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "toggle_panel") {
        togglePanel();
    } else if (request.action === "update_threshold") {
        currentThreshold = request.value;
    }
});

// Nettoyage JSON Gerrit
function cleanGerritResponse(text) {
    if (text.startsWith(")]}'")) return text.substring(4).trim();
    return text;
}

// Fonction principale
async function analyzeChange() {
    const btn = document.getElementById("bp-analyze-btn");
    const resultDiv = document.getElementById("bp-result-panel");
    
    // Récupération ID
    const match = window.location.pathname.match(/(\d+)(?:\/?$|\/?\?)/);
    if (!match) return;
    const changeId = match[1];

    // UI Loading
    btn.textContent = "Analyzing...";
    btn.disabled = true;
    resultDiv.style.display = "none";

    try {
        // 1. Récupération Données Gerrit
        const gerritUrl = `/changes/${changeId}/detail?o=ALL_REVISIONS&o=CURRENT_COMMIT&o=CURRENT_FILES&o=MESSAGES&o=DETAILED_LABELS&o=DETAILED_ACCOUNTS`;
        const gerritResponse = await fetch(gerritUrl);
        const gerritText = await gerritResponse.text();
        const changeJson = JSON.parse(cleanGerritResponse(gerritText));

        // 2. Appel Backend Python
        const predictResponse = await fetch(BACKEND_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(changeJson)
        });

        const prediction = await predictResponse.json();
        
        // 3. Affichage
        displayResult(prediction);

    } catch (error) {
        console.error(error);
        resultDiv.style.display = "block";
        resultDiv.innerHTML = `<div style="color: #d32f2f; font-size: 11px;">Error: ${error.message}. Is app.py running?</div>`;
    } finally {
        btn.textContent = "Analyze Eligibility";
        btn.disabled = false;
    }
}

// AFFICHAGE (Style Pro & Neutre) 
function displayResult(data) {
    const resultDiv = document.getElementById("bp-result-panel");
    const prob = (data.probability * 100).toFixed(1);
    
    // Couleurs indicatives (juste pour la lisibilité, pas de jugement)
    // Vert si > seuil, Orange si proche, Rouge si bas
    let barColor = "#d32f2f"; // Rouge
    if (data.probability >= currentThreshold) barColor = "#2e7d32"; // Vert
    else if (data.probability >= 0.40) barColor = "#f57c00"; // Orange

    const aiText = data.ai_explanation || "No analysis available.";

    resultDiv.innerHTML = `
        <!-- 1. SCORE (Barre de progression) -->
        <div style="margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                <span style="font-weight: 700; color: #555;">Confidence Score</span>
                <span style="font-weight: 800; color: ${barColor};">${prob}%</span>
            </div>
            <div style="width: 100%; height: 6px; background: #eee; border-radius: 3px; overflow: hidden;">
                <div style="width: ${prob}%; height: 100%; background: ${barColor};"></div>
            </div>
        </div>

        <!-- 2. ANALYSE IA (Style clean) -->
        <div style="margin-bottom: 15px; background: #f9f9f9; padding: 10px; border-radius: 4px; border-left: 3px solid #ccc;">
            <div style="font-size: 10px; font-weight: 700; color: #666; margin-bottom: 4px; text-transform: uppercase;">
                AI Insight
            </div>
            <div style="font-size: 11px; line-height: 1.4; color: #333;">
                ${aiText}
            </div>
        </div>

        <!-- 3. DETAILS TECHNIQUES (Grille) -->
        <div style="border-top: 1px solid #eee; padding-top: 10px;">
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 11px;">
                <div>
                    <span style="color: #777;">Risk Depth</span><br>
                    <span style="font-weight: 600;">${data.features_used.max_path_depth}</span>
                </div>
                <div>
                    <span style="color: #777;">Author Trust</span><br>
                    <span style="font-weight: 600;">${parseFloat(data.features_used.author_trust).toFixed(2)}</span>
                </div>
                <div>
                    <span style="color: #777;">Type</span><br>
                    <span style="font-weight: 600;">${data.features_used.nlp_type}</span>
                </div>
                <div>
                    <span style="color: #777;">Entropy</span><br>
                    <span style="font-weight: 600;">${parseFloat(data.features_used.entropy).toFixed(2)}</span>
                </div>
            </div>
        </div>
    `;
    
    resultDiv.style.display = "block";
}

//GESTION DU PANNEAU FLOTTANT 
function togglePanel() {
    const panel = document.getElementById("bp-panel");
    if (panel) {
        panel.style.display = (panel.style.display === "none") ? "block" : "none";
    } else {
        injectUI();
    }
}

function injectUI() {
    if (document.getElementById("bp-panel")) return;

    const panel = document.createElement("div");
    panel.id = "bp-panel";
    
    // Style du panneau 
    panel.style.cssText = `
        position: fixed; bottom: 20px; right: 20px; width: 300px;
        background: white; 
        box-shadow: 0 4px 20px rgba(0,0,0,0.12); 
        border-radius: 8px; border: 1px solid #e0e0e0;
        z-index: 9999; font-family: "Segoe UI", Roboto, sans-serif; font-size: 12px;
        overflow: hidden;
    `;

    // HEADER
    const header = document.createElement("div");
    header.style.cssText = `
        padding: 12px 15px; background: #fff; border-bottom: 1px solid #f0f0f0;
        display: flex; justify-content: space-between; align-items: center;
    `;
    
    header.innerHTML = `
        <span style="font-weight: 700; color: #005c9c; font-size: 13px;">BackportCheck</span>
        <span id="bp-close" style="cursor: pointer; font-size: 18px; color: #999; line-height: 1;">&times;</span>
    `;
    
    // BODY
    const body = document.createElement("div");
    body.style.padding = "15px";

    const btn = document.createElement("button");
    btn.id = "bp-analyze-btn";
    btn.textContent = "Analyze Eligibility";
    btn.style.cssText = `
        width: 100%; padding: 9px; background: #005c9c; color: white;
        border: none; border-radius: 4px; cursor: pointer; font-weight: 600; font-size: 12px;
        transition: background 0.2s;
    `;
    btn.onmouseover = () => btn.style.background = "#004a7c";
    btn.onmouseout = () => btn.style.background = "#005c9c";
    btn.onclick = analyzeChange;

    const resultDiv = document.createElement("div");
    resultDiv.id = "bp-result-panel";
    resultDiv.style.cssText = "margin-top: 15px; display: none;";

    body.appendChild(btn);
    body.appendChild(resultDiv);
    panel.appendChild(header);
    panel.appendChild(body);
    
    document.body.appendChild(panel);

    document.getElementById("bp-close").onclick = () => { panel.style.display = "none"; };
}

// Injection initiale
setTimeout(injectUI, 1500);

// Gestion navigation SPA (Gerrit ne recharge pas la page)
let lastUrl = location.href;
new MutationObserver(() => {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    injectUI();
    const resDiv = document.getElementById("bp-result-panel");
    if (resDiv) resDiv.style.display = "none"; // Reset résultat sur changement de page
  }
}).observe(document, {subtree: true, childList: true});