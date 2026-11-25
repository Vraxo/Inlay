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

    function getActiveView(editor) {
        if (editor.viewMode === "crop")
            return { element: editor.canvas, width: editor.width, height: editor.height };

        return { element: editor.image, width: editor.sourceWidth, height: editor.sourceHeight };
    }

    function getResizeHandleAtPoint(selection, clientX, clientY) {
        const bounds = selection.getBoundingClientRect();
        const hitSize = 14;

        const leftDistance = Math.abs(clientX - bounds.left);
        const rightDistance = Math.abs(clientX - bounds.right);
        const topDistance = Math.abs(clientY - bounds.top);
        const bottomDistance = Math.abs(clientY - bounds.bottom);

        const nearLeft = leftDistance <= hitSize;
        const nearRight = rightDistance <= hitSize;
        const nearTop = topDistance <= hitSize;
        const nearBottom = bottomDistance <= hitSize;

        const horizontal = nearLeft && (!nearRight || leftDistance <= rightDistance)
            ? "w"
            : nearRight ? "e" : "";
        const vertical = nearTop && (!nearBottom || topDistance <= bottomDistance)
            ? "n"
            : nearBottom ? "s" : "";

        if (vertical && horizontal)
            return vertical + horizontal;
        if (vertical)
            return vertical;
        if (horizontal)
            return horizontal;

        return "move";
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
        editor.drawPixelGrid?.();
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
                if (event.button !== 0 || editor.drag || editor.spacePressed)
                    return;

                const grip = event.target.closest("[data-handle]");
                if (!grip && !editor.selection.contains(event.target))
                    return;

                event.preventDefault();
                const bounds = editor.image.getBoundingClientRect();
                if (!bounds.width || !bounds.height)
                    return;

                const handle = grip?.dataset.handle ??
                    getResizeHandleAtPoint(editor.selection, event.clientX, event.clientY);

                editor.drag = {
                    pointerId: event.pointerId,
                    handle,
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
                if (!drag) {
                    const grip = event.target.closest("[data-handle]");
                    const handle = grip?.dataset.handle ??
                        getResizeHandleAtPoint(editor.selection, event.clientX, event.clientY);
                    editor.selection.dataset.edgeHover = handle === "move" ? "" : handle;
                    return;
                }

                if (drag.pointerId !== event.pointerId)
                    return;

                event.preventDefault();
                const deltaX = Math.round((event.clientX - drag.startClientX) * editor.sourceWidth / drag.bounds.width);
                const deltaY = Math.round((event.clientY - drag.startClientY) * editor.sourceHeight / drag.bounds.height);
                applyDelta(editor, drag.handle, deltaX, deltaY);
            };

            editor.onSelectionLeave = () => {
                delete editor.selection.dataset.edgeHover;
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

            editor.viewport = stage.closest(".crop-viewport");
            editor.pixelGrid = stage.closest(".crop-viewport").querySelector(".pixel-grid-overlay");
            editor.zoomControls = Array.from(stage.closest(".panel").querySelectorAll("[data-zoom]"));
            editor.viewControls = Array.from(stage.closest(".panel").querySelectorAll("[data-view-mode]"));
            editor.viewMode = "original";
            editor.zoomFactor = 1;
            editor.panX = 0;
            editor.panY = 0;
            editor.spacePressed = false;
            editor.pointerInside = false;
            editor.panDrag = null;
            editor.fitScale = 0;

            editor.drawPixelGrid = () => {
                const grid = editor.pixelGrid;
                const viewport = editor.viewport;
                if (!grid || !viewport)
                    return;

                const viewportBounds = viewport.getBoundingClientRect();
                const dpr = window.devicePixelRatio || 1;
                const pixelWidth = Math.max(1, Math.round(viewport.clientWidth * dpr));
                const pixelHeight = Math.max(1, Math.round(viewport.clientHeight * dpr));

                if (grid.width !== pixelWidth)
                    grid.width = pixelWidth;
                if (grid.height !== pixelHeight)
                    grid.height = pixelHeight;

                const context = grid.getContext("2d");
                context.setTransform(1, 0, 0, 1, 0, 0);
                context.clearRect(0, 0, pixelWidth, pixelHeight);

                const view = getActiveView(editor);
                if (!view.element.clientWidth || !view.width || !view.height)
                    return;

                const bounds = view.element.getBoundingClientRect();
                const sourcePixelWidth = bounds.width / view.width;
                const sourcePixelHeight = bounds.height / view.height;

                let left = bounds.left - viewportBounds.left;
                let top = bounds.top - viewportBounds.top;
                let width = bounds.width;
                let height = bounds.height;

                if (editor.viewMode === "original") {
                    left += editor.x * sourcePixelWidth;
                    top += editor.y * sourcePixelHeight;
                    width = editor.width * sourcePixelWidth;
                    height = editor.height * sourcePixelHeight;
                }

                const right = left + width;
                const bottom = top + height;
                if (right <= 0 || bottom <= 0 ||
                    left >= viewportBounds.width || top >= viewportBounds.height)
                    return;

                const scale = dpr;
                const x = left * scale;
                const y = top * scale;
                const w = width * scale;
                const h = height * scale;
                const radius = editor.viewMode === "crop"
                    ? editor.radius * Math.min(sourcePixelWidth, sourcePixelHeight)
                    : 0;
                const dark = document.documentElement.dataset.theme === "dark";

                // Grid lines are tied to source-pixel boundaries, not a scaled CSS texture.
                if (sourcePixelWidth >= 3 && sourcePixelHeight >= 3) {
                    context.save();
                    context.beginPath();
                    if (radius > 0)
                        context.roundRect(x, y, w, h, radius * scale);
                    else
                        context.rect(x, y, w, h);
                    context.clip();

                    context.strokeStyle = dark
                        ? "rgba(255,255,255,.48)"
                        : "rgba(25,25,25,.42)";
                    context.lineWidth = dpr;
                    context.beginPath();

                    const firstColumn = Math.max(1, Math.ceil(-left / sourcePixelWidth));
                    const lastColumn = Math.min(
                        editor.width - 1,
                        Math.floor((viewportBounds.width - left) / sourcePixelWidth));
                    for (let column = firstColumn; column <= lastColumn; column++) {
                        const lineX = (left + column * sourcePixelWidth) * scale;
                        context.moveTo(lineX, Math.max(0, y));
                        context.lineTo(lineX, Math.min(pixelHeight, bottom * scale));
                    }

                    const firstRow = Math.max(1, Math.ceil(-top / sourcePixelHeight));
                    const lastRow = Math.min(
                        editor.height - 1,
                        Math.floor((viewportBounds.height - top) / sourcePixelHeight));
                    for (let row = firstRow; row <= lastRow; row++) {
                        const lineY = (top + row * sourcePixelHeight) * scale;
                        context.moveTo(Math.max(0, x), lineY);
                        context.lineTo(Math.min(pixelWidth, right * scale), lineY);
                    }

                    context.stroke();
                    context.restore();
                }

                // Draw the boundary entirely outside the crop. Its inner edge
                // touches the included pixels without covering or separating them.
                const drawOutsideStroke = (lineWidth, color) => {
                    const inset = lineWidth / 2;
                    const outerLeft = x - inset;
                    const outerTop = y - inset;
                    const outerWidth = w + inset * 2;
                    const outerHeight = h + inset * 2;
                    const outerRadius = Math.min(
                        radius * scale + inset,
                        outerWidth / 2,
                        outerHeight / 2);

                    context.beginPath();
                    if (outerRadius > 0)
                        context.roundRect(
                            outerLeft,
                            outerTop,
                            outerWidth,
                            outerHeight,
                            outerRadius);
                    else
                        context.rect(
                            outerLeft,
                            outerTop,
                            outerWidth,
                            outerHeight);

                    context.strokeStyle = color;
                    context.lineWidth = lineWidth;
                    context.stroke();
                };

                context.save();
                context.lineJoin = "round";
                drawOutsideStroke(
                    5 * dpr,
                    dark ? "rgba(255,255,255,.92)" : "rgba(0,0,0,.9)");
                drawOutsideStroke(
                    2.5 * dpr,
                    dark ? "#ff9acb" : "#a82c70");
                context.restore();
            };

            editor.updateTransform = () => {
                stage.style.transform = `translate3d(${editor.panX}px, ${editor.panY}px, 0) scale(${editor.zoomFactor})`;
                const sourceZoom = editor.zoomFactor * editor.fitScale;
                const inverseZoom = editor.zoomFactor > 0 ? 1 / editor.zoomFactor : 1;
                stage.style.setProperty("--inverse-stage-zoom", String(inverseZoom));
                stage.classList.toggle("pixel-inspection-mode", sourceZoom >= 3);
                editor.drawPixelGrid();
            };

            editor.updateZoomButtons = () => {
                for (const button of editor.zoomControls) {
                    const mode = button.dataset.zoom;
                    const selected = mode === "fit"
                        ? Math.abs(editor.zoomFactor - 1) < 0.01
                        : Math.abs(editor.zoomFactor - Number(mode) / editor.fitScale) < 0.03;
                    button.setAttribute("aria-pressed", String(selected));
                }
            };

            editor.resetZoom = () => {
                const view = getActiveView(editor);
                if (!view.element.clientWidth || !view.width)
                    return;
                editor.fitScale = view.element.clientWidth / view.width;
                editor.zoomFactor = 1;
                editor.panX = 0;
                editor.panY = 0;
                editor.updateTransform();
                editor.updateZoomButtons();
            };

            editor.setViewMode = mode => {
                if (mode !== "original" && mode !== "crop" || editor.viewMode === mode)
                    return;

                editor.viewMode = mode;
                stage.classList.toggle("show-crop-result", mode === "crop");
                for (const button of editor.viewControls)
                    button.setAttribute("aria-pressed", String(button.dataset.viewMode === mode));

                editor.resetZoom();
            };

            editor.zoomAt = (nextZoom, clientX, clientY) => {
                if (!editor.fitScale || !editor.viewport)
                    return;
                const viewportBounds = editor.viewport.getBoundingClientRect();
                const oldZoom = editor.zoomFactor;
                const maximumZoom = 8 / editor.fitScale;
                nextZoom = clamp(nextZoom, 1, maximumZoom);
                const centerX = viewportBounds.left + viewportBounds.width / 2;
                const centerY = viewportBounds.top + viewportBounds.height / 2;
                const localX = (clientX - centerX - editor.panX) / oldZoom;
                const localY = (clientY - centerY - editor.panY) / oldZoom;
                editor.panX = clientX - centerX - localX * nextZoom;
                editor.panY = clientY - centerY - localY * nextZoom;
                editor.zoomFactor = nextZoom;
                editor.updateTransform();
                editor.updateZoomButtons();
            };

            editor.setZoom = mode => {
                if (!editor.fitScale)
                    return;
                const viewportBounds = editor.viewport.getBoundingClientRect();
                if (mode === "fit") {
                    editor.resetZoom();
                    return;
                }
                const target = Number(mode);
                if (!Number.isFinite(target) || target <= 0)
                    return;
                editor.zoomAt(target / editor.fitScale,
                    viewportBounds.left + viewportBounds.width / 2,
                    viewportBounds.top + viewportBounds.height / 2);
            };

            editor.onWheel = event => {
                event.preventDefault();
                editor.pointerInside = true;
                const multiplier = event.deltaY < 0 ? 1.2 : 1 / 1.2;
                editor.zoomAt(editor.zoomFactor * multiplier, event.clientX, event.clientY);
            };

            editor.onViewportPointerDown = event => {
                const shouldPan = editor.spacePressed && event.button === 0 || event.button === 1;
                if (!shouldPan || editor.panDrag)
                    return;
                event.preventDefault();
                event.stopPropagation();
                editor.panDrag = {
                    pointerId: event.pointerId,
                    startX: event.clientX,
                    startY: event.clientY,
                    panX: editor.panX,
                    panY: editor.panY
                };
                editor.viewport.classList.add("is-panning");
                editor.viewport.setPointerCapture(event.pointerId);
            };

            editor.onViewportPointerMove = event => {
                editor.pointerInside = true;
                if (!editor.panDrag || editor.panDrag.pointerId !== event.pointerId)
                    return;
                editor.panX = editor.panDrag.panX + event.clientX - editor.panDrag.startX;
                editor.panY = editor.panDrag.panY + event.clientY - editor.panDrag.startY;
                editor.updateTransform();
            };

            editor.onViewportPointerUp = event => {
                if (!editor.panDrag || editor.panDrag.pointerId !== event.pointerId)
                    return;
                editor.panDrag = null;
                editor.viewport.classList.remove("is-panning");
                if (editor.viewport.hasPointerCapture(event.pointerId))
                    editor.viewport.releasePointerCapture(event.pointerId);
            };

            editor.onViewportEnter = () => {
                editor.pointerInside = true;
                if (editor.spacePressed)
                    editor.viewport.classList.add("pan-ready");
            };
            editor.onViewportLeave = () => {
                editor.pointerInside = false;
                editor.viewport.classList.remove("pan-ready");
            };

            editor.onSpaceDown = event => {
                if (event.code !== "Space" || event.target instanceof HTMLElement &&
                    (event.target.isContentEditable || event.target.closest("button, a, input, textarea, select")))
                    return;
                editor.spacePressed = true;
                if (editor.pointerInside) {
                    event.preventDefault();
                    editor.viewport.classList.add("pan-ready");
                }
            };
            editor.onSpaceUp = event => {
                if (event.code !== "Space")
                    return;
                editor.spacePressed = false;
                editor.viewport.classList.remove("pan-ready");
            };
            editor.onWindowBlur = () => {
                editor.spacePressed = false;
                editor.panDrag = null;
                editor.viewport.classList.remove("pan-ready", "is-panning");
            };
            editor.onWindowResize = () => {
                const sourceZoom = editor.zoomFactor * editor.fitScale;
                editor.resetZoom();
                if (sourceZoom > 1.01)
                    editor.setZoom(String(Math.min(8, sourceZoom)));
            };
            editor.onImageLoad = () => {
                drawPreview(editor);
                editor.resetZoom();
            };

            editor.viewport.addEventListener("wheel", editor.onWheel, { passive: false });
            editor.viewport.addEventListener("pointerdown", editor.onViewportPointerDown);
            editor.viewport.addEventListener("pointermove", editor.onViewportPointerMove);
            editor.viewport.addEventListener("pointerup", editor.onViewportPointerUp);
            editor.viewport.addEventListener("pointercancel", editor.onViewportPointerUp);
            editor.viewport.addEventListener("pointerenter", editor.onViewportEnter);
            editor.viewport.addEventListener("pointerleave", editor.onViewportLeave);
            editor.zoomControls.forEach(button => button.addEventListener("click", () => editor.setZoom(button.dataset.zoom)));
            editor.viewControls.forEach(button => button.addEventListener("click", () => editor.setViewMode(button.dataset.viewMode)));
            window.addEventListener("keydown", editor.onSpaceDown);
            window.addEventListener("keyup", editor.onSpaceUp);
            window.addEventListener("blur", editor.onWindowBlur);
            window.addEventListener("resize", editor.onWindowResize);
            editor.onRadiusInput = () => {
                setCrop(editor, editor.x, editor.y, editor.width, editor.height, Number(editor.radiusControl.value));
            };
            editor.onRadiusChange = () => commit(editor);
            editor.selection.addEventListener("pointerdown", editor.onPointerDown);
            editor.selection.addEventListener("pointermove", editor.onPointerMove);
            editor.selection.addEventListener("pointerleave", editor.onSelectionLeave);
            editor.selection.addEventListener("pointerup", editor.onPointerUp);
            editor.selection.addEventListener("pointercancel", editor.onPointerUp);
            editor.selection.addEventListener("keydown", editor.onKeyDown);
            editor.radiusControl.addEventListener("input", editor.onRadiusInput);
            editor.radiusControl.addEventListener("change", editor.onRadiusChange);
            editor.image.addEventListener("load", editor.onImageLoad);

            editors.set(stage, editor);
        }

        const sourceChanged = editor.sourceUrl !== image.src;
        const previousCrop = `${editor.x},${editor.y},${editor.width},${editor.height},${editor.radius}`;
        editor.image = image;
        editor.sourceUrl = image.src;
        editor.selection = selection;
        editor.canvas = canvas;
        editor.readout = readout;
        editor.radiusControl = radiusControl;
        editor.radiusValue = radiusValue;
        editor.dotnet = dotnet;
        editor.sourceWidth = sourceWidth;
        editor.sourceHeight = sourceHeight;
        editor.viewport = stage.closest(".crop-viewport");

        if (sourceChanged) {
            editor.viewMode = "original";
            stage.classList.remove("show-crop-result");
            for (const button of editor.viewControls)
                button.setAttribute("aria-pressed", String(button.dataset.viewMode === "original"));
        }

        setCrop(editor, x, y, width, height, radius);
        const cropChanged = previousCrop !== `${editor.x},${editor.y},${editor.width},${editor.height},${editor.radius}`;
        if (image.complete && image.naturalWidth &&
            (sourceChanged || !editor.fitScale || editor.viewMode === "crop" && cropChanged))
            editor.resetZoom();
    }

    function setZoom(stage, mode) {
        const editor = editors.get(stage);
        editor?.setZoom(mode);
    }

    window.inlayCropEditor = { initialize, setZoom };
})();
