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

 
(() => {
    const editors = new WeakMap();

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function setCrop(editor, x, y, width, height, radius) {
        editor.x = clamp(Math.round(x), 0, editor.sourceWidth - 1);
        editor.y = clamp(Math.round(y), 0, editor.sourceHeight - 1);
        editor.width = clamp(Math.round(width), 1, editor.sourceWidth - editor.x);
        editor.height = clamp(Math.round(height), 1, editor.sourceHeight - editor.y);
        editor.radius = clamp(Math.round(radius), 0, Math.floor(Math.min(editor.width, editor.height) / 2));

        editor.selection.style.left = `${editor.x / editor.sourceWidth * 100}%`;
        editor.selection.style.top = `${editor.y / editor.sourceHeight * 100}%`;
        editor.selection.style.width = `${editor.width / editor.sourceWidth * 100}%`;
        editor.selection.style.height = `${editor.height / editor.sourceHeight * 100}%`;

        editor.radiusControl.max = String(Math.floor(Math.min(editor.width, editor.height) / 2));
        editor.radiusControl.value = String(editor.radius);
        editor.radiusValue.textContent = `${editor.radius} px`;
        editor.readout.textContent = `X ${editor.x} · Y ${editor.y} · ${editor.width} × ${editor.height} px`;

        drawPreview(editor);
    }

    function drawPreview(editor) {
        if (!editor.image.complete || editor.image.naturalWidth === 0)
            return;

        const scale = Math.min(
            1,
            1200 / editor.width,
            1200 / editor.height,
            Math.sqrt(1400000 / (editor.width * editor.height)));

        const outputWidth = Math.max(1, Math.round(editor.width * scale));
        const outputHeight = Math.max(1, Math.round(editor.height * scale));

        if (editor.canvas.width !== outputWidth)
            editor.canvas.width = outputWidth;
        if (editor.canvas.height !== outputHeight)
            editor.canvas.height = outputHeight;

        const context = editor.canvas.getContext("2d");
        context.clearRect(0, 0, outputWidth, outputHeight);
        const radius = Math.min(editor.radius * scale, outputWidth / 2, outputHeight / 2);

        context.save();
        context.beginPath();
        context.roundRect(0, 0, outputWidth, outputHeight, radius);
        context.clip();
        context.drawImage(
            editor.image,
            editor.x,
            editor.y,
            editor.width,
            editor.height,
            0,
            0,
            outputWidth,
            outputHeight);
        context.restore();
    }

    function applyDelta(editor, handle, deltaX, deltaY) {
        const start = editor.drag.start;
        let x = start.x;
        let y = start.y;
        let width = start.width;
        let height = start.height;

        if (handle === "move") {
            x = clamp(start.x + deltaX, 0, editor.sourceWidth - start.width);
            y = clamp(start.y + deltaY, 0, editor.sourceHeight - start.height);
        } else {
            if (handle.includes("w")) {
                const right = start.x + start.width;
                x = clamp(start.x + deltaX, 0, right - 1);
                width = right - x;
            }
            if (handle.includes("e"))
                width = clamp(start.width + deltaX, 1, editor.sourceWidth - start.x);

            if (handle.includes("n")) {
                const bottom = start.y + start.height;
                y = clamp(start.y + deltaY, 0, bottom - 1);
                height = bottom - y;
            }
            if (handle.includes("s"))
                height = clamp(start.height + deltaY, 1, editor.sourceHeight - start.y);
        }

        setCrop(editor, x, y, width, height, editor.radius);
    }

    function commit(editor) {
        editor.dotnet.invokeMethodAsync(
            "CommitCropFromEditor",
            editor.x,
            editor.y,
            editor.width,
            editor.height,
            editor.radius).catch(() => {});
    }

    function initialize(
        stage,
        image,
        selection,
        canvas,
        readout,
        radiusControl,
        radiusValue,
        dotnet,
        sourceWidth,
        sourceHeight,
        x,
        y,
        width,
        height,
        radius) {
        let editor = editors.get(stage);

        if (!editor) {
            editor = {
                stage,
                image,
                selection,
                canvas,
                readout,
                radiusControl,
                dotnet,
                sourceWidth,
                sourceHeight,
                drag: null
            };

            editor.onPointerDown = event => {
                if (event.button !== 0 || editor.drag)
                    return;

                const handle = event.target.closest("[data-handle]");
                if (!handle && !editor.selection.contains(event.target))
                    return;

                event.preventDefault();
                const bounds = editor.image.getBoundingClientRect();
                if (!bounds.width || !bounds.height)
                    return;

                editor.drag = {
                    pointerId: event.pointerId,
                    handle: handle?.dataset.handle || "move",
                    startClientX: event.clientX,
                    startClientY: event.clientY,
                    bounds,
                    start: {
                        x: editor.x,
                        y: editor.y,
                        width: editor.width,
                        height: editor.height
                    }
                };

                editor.selection.setPointerCapture(event.pointerId);
            };

            editor.onPointerMove = event => {
                const drag = editor.drag;
                if (!drag || drag.pointerId !== event.pointerId)
                    return;

                event.preventDefault();
                const deltaX = Math.round((event.clientX - drag.startClientX) * editor.sourceWidth / drag.bounds.width);
                const deltaY = Math.round((event.clientY - drag.startClientY) * editor.sourceHeight / drag.bounds.height);
                applyDelta(editor, drag.handle, deltaX, deltaY);
            };

            editor.onPointerUp = event => {
                if (!editor.drag || editor.drag.pointerId !== event.pointerId)
                    return;

                editor.drag = null;
                if (editor.selection.hasPointerCapture(event.pointerId))
                    editor.selection.releasePointerCapture(event.pointerId);
                commit(editor);
            };

            editor.onKeyDown = event => {
                if (!["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key))
                    return;

                event.preventDefault();
                const step = event.shiftKey ? 10 : 1;
                const handle = event.target.closest("[data-handle]")?.dataset.handle || "move";
                const deltaX = event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : 0;
                const deltaY = event.key === "ArrowUp" ? -step : event.key === "ArrowDown" ? step : 0;

                editor.drag = {
                    start: {
                        x: editor.x,
                        y: editor.y,
                        width: editor.width,
                        height: editor.height
                    }
                };
                applyDelta(editor, handle, deltaX, deltaY);
                editor.drag = null;
                commit(editor);
            };

            editor.onRadiusInput = () => {
                setCrop(editor, editor.x, editor.y, editor.width, editor.height, Number(editor.radiusControl.value));
            };
            editor.onRadiusChange = () => commit(editor);
            editor.onImageLoad = () => drawPreview(editor);

            editor.selection.addEventListener("pointerdown", editor.onPointerDown);
            editor.selection.addEventListener("pointermove", editor.onPointerMove);
            editor.selection.addEventListener("pointerup", editor.onPointerUp);
            editor.selection.addEventListener("pointercancel", editor.onPointerUp);
            editor.selection.addEventListener("keydown", editor.onKeyDown);
            editor.radiusControl.addEventListener("input", editor.onRadiusInput);
            editor.radiusControl.addEventListener("change", editor.onRadiusChange);
            editor.image.addEventListener("load", editor.onImageLoad);

            editors.set(stage, editor);
        }

        editor.image = image;
        editor.selection = selection;
        editor.canvas = canvas;
        editor.readout = readout;
        editor.radiusControl = radiusControl;
        editor.radiusValue = radiusValue;
        editor.dotnet = dotnet;
        editor.sourceWidth = sourceWidth;
        editor.sourceHeight = sourceHeight;

        setCrop(editor, x, y, width, height, radius);
    }

    window.inlayCropEditor = { initialize };
})();
