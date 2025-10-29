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


    // --- 3️⃣ Ensure keyboard focus works (important for gameplay) ---
    const observer = new MutationObserver(() => {
      const canvas = this.$refs.unityContainer.querySelector("canvas");
      if (canvas && !canvas.hasAttribute("tabindex")) {
        canvas.setAttribute("tabindex", "0");
        canvas.focus();
      }
    });
    observer.observe(this.$refs.unityContainer, { childList: true, subtree: true });
  },
  beforeUnmount() {
    // --- 4️⃣ Clean up the <base> tag --- 
    if (this.baseElement) {
      document.head.removeChild(this.baseElement);
    }
  }
};
</script>

<style scoped>
#unity-container {
  width: 100%;
  height: 100vh;
  background: #000;
  overflow: hidden;
}
</style>
