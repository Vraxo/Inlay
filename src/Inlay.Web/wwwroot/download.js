window.inlay ??= {};

window.inlay.download = async (fileName, streamReference, contentType = "image/png") => {
    const stream = await streamReference.arrayBuffer();
    const blob = new Blob([stream], { type: contentType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = fileName;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
};
