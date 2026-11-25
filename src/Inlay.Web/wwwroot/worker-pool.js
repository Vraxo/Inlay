let workers = [];
let queue = [];
let nextRequestId = 0;
let progressTarget = null;
let initialization = null;
let disposed = false;

export async function initialize(maxWorkers, dotNetProgressTarget) {
    if (initialization)
        return initialization;

    disposed = false;
    progressTarget = dotNetProgressTarget;

    const cores = Math.max(1, Number(navigator.hardwareConcurrency) || 2);
    const memory = Number(navigator.deviceMemory);
    const canRunTwo = cores >= 4 && (!Number.isFinite(memory) || memory >= 4);
    const desiredWorkers = Math.max(1, Math.min(2, Number(maxWorkers) || 1, canRunTwo ? 2 : 1));

    initialization = (async () => {
        await createWorker();
        if (desiredWorkers > 1) {
            try {
                await createWorker();
            }
            catch {
                // Retain one ready worker on memory-constrained devices.
            }
        }

        if (workers.length === 0)
            throw new Error("No background image worker could be started.");

        return workers.length;
    })();

    return initialization;
}

function createWorker() {
    return new Promise((resolve, reject) => {
        const worker = new Worker(new URL("./processing-worker.js", import.meta.url), { type: "module" });
        const slot = { worker, ready: false, job: null };
        const timeout = setTimeout(() => {
            if (slot.ready)
                return;
            worker.terminate();
            reject(new Error("The background image worker did not start in time."));
        }, 60000);

        worker.addEventListener("message", event => {
            const message = event.data;

            if (message.type === "ready") {
                if (slot.ready)
                    return;
                slot.ready = true;
                clearTimeout(timeout);
                workers.push(slot);
                resolve();
                dispatch();
                return;
            }

            if (message.type === "startup-error") {
                if (!slot.ready) {
                    clearTimeout(timeout);
                    worker.terminate();
                    reject(new Error(message.error || "The background image worker failed to start."));
                }
                else {
                    failWorker(slot, new Error(message.error || "The background image worker failed."));
                }
                return;
            }

            if (message.type === "progress") {
                if (slot.job && slot.job.requestId === message.requestId && progressTarget) {
                    progressTarget.invokeMethodAsync(
                        "ReportWorkerProgress", slot.job.imageId, message.progress
                    ).catch(() => {});
                }
                return;
            }

            if (!slot.job || slot.job.requestId !== message.requestId)
                return;

            if (message.type === "result") {
                const output = message.output instanceof Uint8Array
                    ? message.output
                    : new Uint8Array(message.output);
                completeJob(slot, null, output);
            }
            else if (message.type === "error") {
                completeJob(slot, new Error(message.error || "Image processing failed."));
            }
        });

        worker.addEventListener("error", event => {
            clearTimeout(timeout);
            const error = new Error(event.message || "The background image worker stopped unexpectedly.");
            if (!slot.ready)
                reject(error);
            else
                failWorker(slot, error);
            worker.terminate();
        });
    });
}

export function process(imageId, sourceBytes) {
    if (!initialization || workers.length === 0 || disposed)
        return Promise.reject(new Error("Background image workers are not ready."));

    return new Promise((resolve, reject) => {
        const bytes = sourceBytes instanceof Uint8Array ? sourceBytes : new Uint8Array(sourceBytes);
        const buffer = bytes.byteOffset === 0 && bytes.byteLength === bytes.buffer.byteLength
            ? bytes.buffer
            : bytes.slice().buffer;

        queue.push({ requestId: ++nextRequestId, imageId, buffer, resolve, reject });
        dispatch();
    });
}

function dispatch() {
    if (disposed)
        return;

    for (const slot of workers) {
        if (!slot.ready || slot.job || queue.length === 0)
            continue;

        const job = queue.shift();
        slot.job = job;
        const sourceBytes = new Uint8Array(job.buffer);

        try {
            slot.worker.postMessage({ requestId: job.requestId, sourceBytes }, [job.buffer]);
        }
        catch (error) {
            completeJob(slot, error instanceof Error ? error : new Error(String(error)));
        }
    }
}

function completeJob(slot, error, result) {
    const job = slot.job;
    slot.job = null;

    if (job) {
        if (error)
            job.reject(error);
        else
            job.resolve(result);
    }
    dispatch();
}

function failWorker(slot, error) {
    const index = workers.indexOf(slot);
    if (index >= 0)
        workers.splice(index, 1);

    if (slot.job) {
        const job = slot.job;
        slot.job = null;
        job.reject(error);
    }

    if (workers.length === 0) {
        for (const job of queue.splice(0))
            job.reject(new Error("All background image workers stopped."));
    }
    else {
        dispatch();
    }
}

export function dispose() {
    disposed = true;
    for (const slot of workers) {
        slot.worker.terminate();
        if (slot.job)
            slot.job.reject(new Error("The batch was interrupted."));
    }
    workers = [];
    for (const job of queue.splice(0))
        job.reject(new Error("The batch was interrupted."));
    progressTarget = null;
    initialization = null;
}
