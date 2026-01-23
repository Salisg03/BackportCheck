document.addEventListener('DOMContentLoaded', () => {
    const slider = document.getElementById('threshold');
    const label = document.getElementById('thresh-val');
    const status = document.getElementById('save-status');
    const toggleBtn = document.getElementById('toggle-panel');
    const BACKEND_THRESHOLD_URL = "http://localhost:5000/threshold";

    // 1. Load saved value
    chrome.storage.local.get(['threshold'], (result) => {
        if (result.threshold) {
            slider.value = result.threshold;
            label.textContent = parseFloat(result.threshold).toFixed(2);
        }
    });

    // 2. Handle Slider (Real-time update)
    slider.addEventListener('input', () => {
        const val = slider.value;
        label.textContent = parseFloat(val).toFixed(2);
        
        // Visual Feedback
        status.classList.add('visible');
        setTimeout(() => status.classList.remove('visible'), 1500);

        // Update Everything
        syncThreshold(val);
    });

    // 3. Handle Button (Syncs + Toggles)
    toggleBtn.addEventListener('click', () => {
        // We sync the threshold again just to be 100% sure the content script has it
        syncThreshold(slider.value);
        
        // Then we toggle
        sendMessageToContent({ action: "toggle_panel" });
        
        // Optional: Close popup after clicking toggle for cleaner UX
        // window.close(); 
    });
    const infoTrigger = document.getElementById('toggle-info');
    const infoSection = document.getElementById('info-section');
    const arrow = document.querySelector('.arrow');

    infoTrigger.addEventListener('click', (e) => {
        e.preventDefault(); // Stop it from jumping to top of page
        
        if (infoSection.style.display === "none") {
            // Show it
            infoSection.style.display = "block";
            arrow.classList.add("open");
            // Optional: Auto-resize popup height if needed (Chrome handles this mostly)
        } else {
            // Hide it
            infoSection.style.display = "none";
            arrow.classList.remove("open");
        }
    });
    // --- Helpers ---

    function syncThreshold(val) {
        // Save to Chrome Storage
        chrome.storage.local.set({ threshold: val });

        // Send to Content Script (Immediate UI update)
        sendMessageToContent({ action: "update_threshold", value: parseFloat(val) });

        // Send to Python Backend (AI Context update)
        fetch(BACKEND_THRESHOLD_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ threshold: val })
        }).catch(err => console.log("Backend offline?", err));
    }

    function sendMessageToContent(message) {
        chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs[0]) {
                chrome.tabs.sendMessage(tabs[0].id, message);
            }
        });
    }
});