!<template>
  <div class="webcam-wrapper">
    <img src="https://images.unsplash.com/photo-1761872936185-4ece7c1128ab?ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&q=80&w=1170" alt="" id="webcam"/>
    <p>Video Stream unavailable <br/>:(</p>
  </div>
</template>

<script>
  export default {

  }
</script>

<style scoped>
  .webcam-wrapper {
    width: 100%;
    height: 100%;

    display: grid;
    grid-template-columns: 1fr;
    grid-template-rows: 1fr;

    place-items: center;
    align-content: center;
  }

  .webcam-wrapper > * {
    grid-area: 1 / 1;
  }

  img {
    z-index: 1;
    width: 100%;
    height: 100%;

    object-fit: cover;
  }

  p {
    color: white;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 5.5rem;
    text-align: center;
    font-style: bold;
    font-weight: bolder;
    text-shadow: 0 0 5px #aaaaaa;
  }
</style>
