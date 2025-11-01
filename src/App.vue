<template>
  <div class="app">
    <SubwaySurfers/>
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
</style>
