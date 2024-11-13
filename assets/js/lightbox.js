document.addEventListener("DOMContentLoaded", function () {
  const grid = document.getElementById("image-grid");
  const items = Array.from(grid.getElementsByTagName("figure"));

  // Function to get image dimensions
  function getImageDimensions(img) {
    return new Promise((resolve, reject) => {
      if (img.complete) {
        resolve({ width: img.naturalWidth, height: img.naturalHeight });
      } else {
        img.onload = () =>
          resolve({ width: img.naturalWidth, height: img.naturalHeight });
        img.onerror = reject;
      }
    });
  }

  // Calculate aspect ratios and sort
  Promise.all(
    items.map((item) => {
      const img = item.querySelector("img");
      return getImageDimensions(img).then((dimensions) => {
        const aspectRatio = dimensions.width / dimensions.height;
        item.dataset.aspectRatio = aspectRatio;
        return item;
      });
    }),
  ).then((itemsWithRatios) => {
    itemsWithRatios.sort((a, b) => {
      return (
        parseFloat(b.dataset.aspectRatio) - parseFloat(a.dataset.aspectRatio)
      );
    });
    itemsWithRatios.forEach((item) => grid.appendChild(item));
  });
});

document.addEventListener("DOMContentLoaded", function () {
  const gallery = document.querySelector(".image-gallery");
  if (gallery) {
    gallery.addEventListener("click", function (e) {
      if (e.target.tagName === "IMG") {
        e.preventDefault();
        const lightbox = document.createElement("div");
        lightbox.className = "lightbox";
        const img = document.createElement("img");
        img.src = e.target.parentElement.href;
        lightbox.appendChild(img);
        document.body.appendChild(lightbox);
        lightbox.addEventListener("click", function () {
          document.body.removeChild(lightbox);
        });
      }
    });
  }
});
