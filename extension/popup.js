document.addEventListener('DOMContentLoaded', () => {
    const slider = document.getElementById('threshold');
    const label = document.getElementById('thresh-val');
    const toggleBtn = document.getElementById('toggle-panel');

    // Charger la valeur sauvegardée (si existe)
    chrome.storage.local.get(['threshold'], (result) => {
        if (result.threshold) {
            slider.value = result.threshold;
            label.textContent = result.threshold;
        }
    });

    // Quand on bouge le curseur
    slider.addEventListener('input', () => {
        const val = slider.value;
        label.textContent = val;
        
        // 1. Sauvegarder
        chrome.storage.local.set({ threshold: val });

        // 2. Envoyer à la page active
        sendMessageToContent({ action: "update_threshold", value: parseFloat(val) });
    });

    // Quand on clique sur le bouton Show/Hide
    toggleBtn.addEventListener('click', () => {
        sendMessageToContent({ action: "toggle_panel" });
    });
});

// Fonction utilitaire pour parler à content.js
function sendMessageToContent(message) {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs[0]) {
            chrome.tabs.sendMessage(tabs[0].id, message);
        }
    });
}