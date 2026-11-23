(() => {
    function getDropzone(target) {
        return target instanceof Element ? target.closest(".dropzone") : null;
    }

    document.addEventListener("dragover", event => {
        const dropzone = getDropzone(event.target);
        if (!dropzone || !event.dataTransfer?.types.includes("Files"))
            return;

        event.preventDefault();
        event.dataTransfer.dropEffect = "copy";
        dropzone.classList.add("dragging");
    });

    document.addEventListener("dragleave", event => {
        const dropzone = getDropzone(event.target);
        if (dropzone && (!event.relatedTarget || !dropzone.contains(event.relatedTarget)))
            dropzone.classList.remove("dragging");
    });

    document.addEventListener("drop", event => {
        const dropzone = getDropzone(event.target);
        if (!dropzone)
            return;

        event.preventDefault();
        dropzone.classList.remove("dragging");

        const files = event.dataTransfer?.files;
        const input = dropzone.querySelector('input[type="file"]');
        if (!files?.length || !input)
            return;

        const transfer = new DataTransfer();
        for (const file of files)
            transfer.items.add(file);

        input.files = transfer.files;
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });

    document.addEventListener("dragend", () => {
        document.querySelectorAll(".dropzone.dragging")
            .forEach(dropzone => dropzone.classList.remove("dragging"));
    });
})();
