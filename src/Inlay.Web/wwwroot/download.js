window.inlay = {
    download: async (fileName, streamReference) => {
        const stream = await streamReference.arrayBuffer();
        const blob = new Blob([stream], { type: "image/png" });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = fileName;
        anchor.click();
        anchor.remove();
        URL.revokeObjectURL(url);
    }
};
