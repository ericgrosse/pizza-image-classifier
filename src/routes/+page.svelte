<script>
  import * as tf from '@tensorflow/tfjs';
  import * as mobilenet from '@tensorflow-models/mobilenet';

  let model;
  let imageSrc;
  let predictions = [];
  let loading = false;

  const loadModel = async () => {
    console.log('Loading model...');
    loading = true;
    model = await mobilenet.load();
    loading = false;
    console.log('Model loaded.');
  };

  const predict = async () => {
    loading = true;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = imageSrc;
    await img.decode();
    const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([224, 224]).toFloat();
    const offset = tf.scalar(127.5);
    const normalized = tensor.sub(offset).div(offset);
    const batched = normalized.reshape([1, 224, 224, 3]);
    const result = await model.classify(batched);
    predictions = result;
    loading = false;
  };

  $: if (imageSrc) predict();
</script>

<style>
  img {
    max-width: 100%;
    height: auto;
  }
</style>

<h1>Image Classifier</h1>

<button on:click={loadModel} disabled={model || loading}>Load Model</button>

{#if loading}
  <p>Loading...</p>
{:else if model}
  <input type="file" accept="image/*" on:change={event => {imageSrc = URL.createObjectURL(event.target.files[0])}}>

  {#if imageSrc}
    <div>
      <img src={imageSrc}>
      {#if predictions.length > 0}
        <ul>
          {#each predictions as {className, probability}}
            <li>{className}: {probability.toFixed(2)}</li>
          {/each}
        </ul>
      {:else}
        <p>Click the "Predict" button to classify the image.</p>
      {/if}
      <button on:click={() => {imageSrc = ''; predictions = [];}}>Clear</button>
      <button on:click={predict}>Predict</button>
    </div>
  {/if}
{/if}
