<template>
  <div class="app">
    <div class="webcam"></div>
    <div class="game">
      <SubwaySurfers/>
    </div>
  </div>
</template>

<script>
import SubwaySurfers from './components/SubwaySurfers.vue';

  export default {
    components: { SubwaySurfers }
  }
</script>

<style scoped>
  .app {
    display: flex;
    padding-block: 5vh;
    padding-inline: 5vh;

    width: 100vw;
    height: 100vh;
    overflow: hidden;

    --background-colour: hsl(216, 28%, 7%);
    --grid-lines-colour: hsl(0, 0%, 15%);
    background-image:
      linear-gradient(var(--grid-lines-colour) 1px, transparent 1px),
      linear-gradient(to right, var(--grid-lines-colour) 1px, var(--background-colour) 1px);
    background-size: 40px 40px;
  }

  .webcam {
    /* background: red; */
    width: 40%;
    height: 100%;
  }

  .game {
    /* background: blue; */
    width: 60%;
    height: 100%;

    border-radius: 8px;
    box-shadow: 0 0 20px #aaaaaa;
    overflow: hidden;
  }
</style>
