const imageUpload = document.getElementById("imageUpload");
const mainContainer = document.getElementById("main-container");
const msgBox = document.getElementById("msg-box");

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
]).then(start);

async function start() {
  const container = document.createElement("div");
  container.classList.add("relative");
  mainContainer.append(container);
  const labeledFaceDescriptors = await loadLabeledImages();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
  let image;
  let canvas;
  msgBox.innerHTML = "Loaded";
  imageUpload.addEventListener("change", async () => {
    if (image) image.remove();
    if (canvas) canvas.remove();
    image = await faceapi.bufferToImage(imageUpload.files[0]);
    image.classList.add("shadow");
    image.classList.add("shadow-white");
    image.classList.add("rounded");
    container.append(image);
    image.width = 720;
    image.height = 560;
    canvas = faceapi.createCanvasFromMedia(image);
    container.append(canvas);
    if (msgBox) msgBox.remove();
    const displaySize = { width: image.width, height: image.height };
    faceapi.matchDimensions(canvas, displaySize);
    const detections = await faceapi
      .detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    const results = resizedDetections.map((d) =>
      faceMatcher.findBestMatch(d.descriptor)
    );
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result.toString(),
      });
      drawBox.draw(canvas);
    });
  });
}

function loadLabeledImages() {
  const labels = [
    {
      name: "Ashutosh Mohapatra",
      images: [
        `https://firebasestorage.googleapis.com/v0/b/firecrud-aa9ec.appspot.com/o/tutu.jpeg?alt=media&token=4ced3453-1e31-4a67-bb27-20d63104af7c`,
      ],
    },
    {
      name: "Ashirbad Panigrahi",
      images: [
        `https://firebasestorage.googleapis.com/v0/b/firecrud-aa9ec.appspot.com/o/gudu-1.jpeg?alt=media&token=43cdfbcc-7a6d-4430-8f72-8262ef755c8e`,
        `https://firebasestorage.googleapis.com/v0/b/firecrud-aa9ec.appspot.com/o/gudu-2.jpeg?alt=media&token=4db833b7-e940-45e9-bdbc-9b2c1883fddb`,
      ],
    },
  ];
  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];
      for (const src of label.images) {
        const img = await faceapi.fetchImage(src);
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }
      return new faceapi.LabeledFaceDescriptors(label.name, descriptions);
    })
  );
}
