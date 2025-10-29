<template>
  <div id="unity-container" ref="unityContainer"></div>
</template>

<script>
export default {
  name: "SubwaySurfers",
  mounted() {
    const basePath = "/subway-surfers-wasm/";

    // --- 0️⃣ Add <base> tag to resolve relative paths correctly ---
    this.baseElement = document.createElement('base');
    this.baseElement.href = basePath;
    document.head.appendChild(this.baseElement);

    // --- 1️⃣ Unity configuration (same as original index.html) ---
    window.config = {
      loader: "unity",
      debug: false,
      maxRatio: 16 / 9,
      minRatio: 9 / 16,
      title: "Subway Surfers: San Francisco",
      unityVersion: "2019.4.18f1",
      unityWebglBuildUrl: "Build/SanFrancisco/SanFrancisco.json", // This can be relative now because of the <base> tag
      fileSize: 35,
      cachedDecompressedFileSizes: {
        "SanFrancisco.asm.code.unityweb": 9077143,
        "SanFrancisco.asm.framework.unityweb": 86369,
        "SanFrancisco.asm.memory.unityweb": 951447,
        "SanFrancisco.data.unityweb": 18323917,
        "SanFrancisco.wasm.code.unityweb": 7279617,
        "SanFrancisco.wasm.framework.unityweb": 90693,
      },
    };

    // --- 2️⃣ Load scripts like the original index.html ---
    const loadScript = (src) => {
      return new Promise((resolve, reject) => {
        const script = document.createElement("script");
        // We load scripts by their absolute path, so they are not affected by the <base> tag.
        script.src = basePath + src;
        script.onload = resolve;
        script.onerror = reject;
        document.body.appendChild(script);
      });
    };

    // Load scripts in the order they appear in the original index.html
    loadScript("4399.z.js")
      .then(() => {
        return loadScript("master-loader.js");
      })
      .catch(err => console.error("Failed to load game scripts:", err));


    // --- 3️⃣ Watch for the game container to be added to the body, then move it ---
    this.observer = new MutationObserver((mutationsList, obs) => {
      for (const mutation of mutationsList) {
        if (mutation.type === 'childList') {
          for (const node of mutation.addedNodes) {
            // The loader script adds a div with id 'game-container' to the document body
            if (node.id === 'game-container') {
              // Move the game container from the body into our Vue component
              this.$refs.unityContainer.appendChild(node);

              // Now that the container is moved, we need to watch for the canvas inside it
              const canvasObserver = new MutationObserver(() => {
                const canvas = node.querySelector('canvas');
                if (canvas) {
                  // Ensure canvas can get keyboard focus for gameplay
                  if (!canvas.hasAttribute('tabindex')) {
                    canvas.setAttribute('tabindex', '0');
                  }
                  canvas.focus();
                  // Stop observing once the canvas is found and configured
                  canvasObserver.disconnect();
                }
              });

              // Start observing the moved #game-container for the canvas to be added
              canvasObserver.observe(node, { childList: true, subtree: true });
              this.canvasObserver = canvasObserver;

              // Stop observing the body, our main job is done
              obs.disconnect();
              return;
            }
          }
        }
      }
    });

    // Start observing the document body for new elements
    this.observer.observe(document.body, { childList: true });
  },
  beforeUnmount() {
    // --- 4️⃣ Clean up the <base> tag and observers --- 
    if (this.baseElement) {
      document.head.removeChild(this.baseElement);
    }
    // Disconnect observers to prevent memory leaks
    if (this.observer) {
      this.observer.disconnect();
    }
    if (this.canvasObserver) {
      this.canvasObserver.disconnect();
    }
  }
};
</script>

<style scoped>
#unity-container {
  width: 100%;
  width: 100%;
  height: 100vh;
  background: #000;
  overflow: hidden;
}
</style>
