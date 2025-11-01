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


    // --- 3️⃣ Watch for game elements to be added to the body, then move them ---
    this.observer = new MutationObserver((mutationsList, obs) => {
      for (const mutation of mutationsList) {
        if (mutation.type === 'childList') {
          for (const node of mutation.addedNodes) {
            // The loader script adds elements like #game-container, #spinner, and #loader to the document body
            if (node.id === 'game-container' || node.id === 'spinner' || node.id === 'loader') {
              // Move the element from the body into our Vue component
              this.$refs.unityContainer.appendChild(node);

              // The #game-container is the main element, which will contain the canvas
              if (node.id === 'game-container') {
                // --- 4️⃣ Inject a style tag to override game's !important styles ---
                const styleTag = document.createElement('style');
                styleTag.id = 'game-style-override';
                styleTag.innerHTML = `
                  #game-container {
                    position: static !important;
                    margin-left: 0 !important;
                    margin-top: 0 !important;
                  }

                  .spinner {
                      transform: translate(28vw, 50vh);
                  }
                `;
                document.head.appendChild(styleTag);
                this.styleTag = styleTag;

                // We still need to watch for the canvas to do one-time setup.
                const canvasObserver = new MutationObserver(() => {
                  const canvas = node.querySelector('canvas');
                  if (canvas) {
                    if (!canvas.hasAttribute('tabindex')) {
                      canvas.setAttribute('tabindex', '0');
                    }
                    canvas.focus();
                    canvasObserver.disconnect();
                  }
                });
                canvasObserver.observe(node, { childList: true, subtree: true });
                this.canvasObserver = canvasObserver;

                // Stop observing the body, our main job is done
                obs.disconnect();
                return;
              }
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
    // Disconnect observers and remove injected styles to prevent memory leaks
    if (this.observer) {
      this.observer.disconnect();
    }
    if (this.canvasObserver) {
      this.canvasObserver.disconnect();
    }
    if (this.styleTag) {
      document.head.removeChild(this.styleTag);
    }
  }
};
</script>

<style>
#unity-container {
  width: 100%;
  /* max-width: 1632px; When this is exceeded, a weird gray screen artifact starts appearing */
  height: 100%;
  background: black;
  position: relative;
}

#spinner {
  position: absolute;
}

#game-container, #game, #canvas {
  width: 100% !important;
  height: 100% !important;
  z-index: 1;
}

:hover, :active, :focus{
  outline: 0;
  outline: none;
}
</style>
