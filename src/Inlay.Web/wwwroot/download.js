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

window.inlay.createObjectUrl = (bytes, contentType = "image/png") =>
    URL.createObjectURL(new Blob([bytes], { type: contentType }));

window.inlay.revokeObjectUrl = url => {
    if (url)
        URL.revokeObjectURL(url);
};

window.inlay.copyText = async text => {
    if (navigator.clipboard?.writeText) {
        try {
            await navigator.clipboard.writeText(text);
            return true;
        } catch {
            // Fall through to the legacy clipboard path for restricted contexts.
        }
    }

    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand("copy");
    textarea.remove();

    if (!copied)
        throw new Error("Clipboard access is unavailable in this browser.");

    return true;
};
