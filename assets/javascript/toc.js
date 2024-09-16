document.addEventListener("DOMContentLoaded", function () {
  const tocList = document.getElementById("toc-list");
  const headers = document.querySelectorAll(
    "article h1, article h2, article h3",
  );

  headers.forEach(function (header) {
    const listItem = document.createElement("li");
    const link = document.createElement("a");
    link.setAttribute("href", "#" + header.id);
    link.textContent = header.textContent;

    listItem.appendChild(link);
    tocList.appendChild(listItem);
  });
});
