// CONFIGURATION 
const BACKEND_URL = "http://localhost:5000/predict";
let currentThreshold = 0.50; 

// Listen for changes from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === "toggle_panel") {
        togglePanel();
    } else if (request.action === "update_threshold") {
        currentThreshold = request.value;
    }
});

// Clean Gerrit JSON response
function cleanGerritResponse(text) {
    if (text.startsWith(")]}'")) return text.substring(4).trim();
    return text;
}

// Main function
async function analyzeChange() {
    const btn = document.getElementById("bp-analyze-btn");
    const resultDiv = document.getElementById("bp-result-panel");
    
    // Get change ID
    const match = window.location.pathname.match(/(\d+)(?:\/?$|\/?\?)/);
    if (!match) return;
    const changeId = match[1];

    // UI Loading
    btn.textContent = "Analyzing...";
    btn.disabled = true;
    resultDiv.style.display = "none";

    try {
        // 1. Fetch Gerrit data
        const gerritUrl = `/changes/${changeId}/detail?o=ALL_REVISIONS&o=CURRENT_COMMIT&o=CURRENT_FILES&o=MESSAGES&o=DETAILED_LABELS&o=DETAILED_ACCOUNTS`;
        const gerritResponse = await fetch(gerritUrl);
        const gerritText = await gerritResponse.text();
        const changeJson = JSON.parse(cleanGerritResponse(gerritText));

        // 2. Call Python backend
        const predictResponse = await fetch(BACKEND_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(changeJson)
        });

        const prediction = await predictResponse.json();
        
        // 3. Display result
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

// DISPLAY (Updated UI with Verdict Banner)
function displayResult(data) {
    const resultDiv = document.getElementById("bp-result-panel");
    const prob = (data.probability * 100).toFixed(1);
    
    // Verdict Logic
    const isRecommended = data.probability >= currentThreshold;
    const color = isRecommended ? "#2e7d32" : "#d32f2f"; 
    const bgColor = isRecommended ? "#e8f5e9" : "#ffebee";
    const verdictText = isRecommended ? "RECOMMENDED" : "NOT RECOMMENDED";
    const verdictIcon = isRecommended ? "✅" : "⛔";
    
    const aiText = data.ai_explanation || "No analysis available.";
    const f = data.features_used; 

    // Helper
    const fmt = (n) => parseFloat(n).toFixed(2);

    resultDiv.innerHTML = `
        <!-- 1. VERDICT BANNER -->
        <div style="background: ${bgColor}; border: 1px solid ${color}; border-radius: 6px; padding: 12px; text-align: center; margin-bottom: 15px;">
            <div style="color: ${color}; font-weight: 800; font-size: 14px; letter-spacing: 0.5px; margin-bottom: 4px;">
                ${verdictIcon} ${verdictText}
            </div>
            <div style="color: ${color}; font-size: 11px; opacity: 0.9;">
                Confidence: <b>${prob}%</b> (Threshold: ${currentThreshold})
            </div>
        </div>

        <!-- 2. AI INSIGHT (Now powered by FULL Context) -->
        <div style="margin-bottom: 15px;">
            <div style="font-size: 11px; font-weight: 700; color: #555; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">
                AI Reasoning
            </div>
            <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; border-left: 3px solid #607d8b; font-size: 12px; line-height: 1.5; color: #333;">
                ${aiText}
            </div>
        </div>

        <!-- 3. KEY METRICS GRID (The "Big 4" Summary) -->
        <div style="border-top: 1px solid #eee; padding-top: 12px;">
            <div style="font-size: 10px; font-weight: 700; color: #999; margin-bottom: 8px;">KEY INDICATORS</div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; font-size: 11px;">
                
                <!-- Factor 1 -->
                <div>
                    <span style="color: #666; cursor: help;" title="Author Trust Score based on past history">👤 Author Trust</span><br>
                    <span style="font-weight: 600; font-size: 13px;">${fmt(f.author_trust)}</span>
                </div>

                <!-- Factor 2 -->
                <div>
                    <span style="color: #666; cursor: help;" title="Probability that files in these directories are backported">📂 File Fit</span><br>
                    <span style="font-weight: 600; font-size: 13px;">${fmt(f.file_risk)}</span>
                </div>

                <!-- Factor 3 -->
                <div>
                    <span style="color: #666; cursor: help;" title="Code Entropy / Complexity">🧩 Complexity</span><br>
                    <span style="font-weight: 600; font-size: 13px;">${fmt(f.entropy)}</span>
                </div>

                <!-- Factor 4 -->
                <div>
                    <span style="color: #666;">🏷️ Type</span><br>
                    <span style="font-weight: 600; font-size: 12px; background: #eee; padding: 2px 6px; border-radius: 4px;">${f.nlp_type}</span>
                </div>
            </div>
        </div>
    `;
    
    resultDiv.style.display = "block";
}
// FLOATING PANEL MANAGEMENT
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
    
    // Panel styling
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

// Initial injection
setTimeout(injectUI, 1500);

// SPA navigation handling (Gerrit doesn't reload the page)
let lastUrl = location.href;
new MutationObserver(() => {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    injectUI();
    const resDiv = document.getElementById("bp-result-panel");
    if (resDiv) resDiv.style.display = "none"; // Reset result on page change
  }
}).observe(document, {subtree: true, childList: true});