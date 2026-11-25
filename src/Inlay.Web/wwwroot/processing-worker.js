import { dotnet } from "./_framework/dotnet.js";

let processImage = null;

try {
    const { getAssemblyExports, getConfig } = await dotnet.create();
    const config = getConfig();
    const exports = await getAssemblyExports(config.mainAssemblyName);
    processImage = exports.Inlay.Web.Workers.ImageProcessingWorker.ProcessImage;
    self.postMessage({ type: "ready" });
}
catch (error) {
    self.postMessage({
        type: "startup-error",
        error: error instanceof Error ? error.message : String(error)
    });
}

self.addEventListener("message", event => {
    const { requestId, sourceBytes } = event.data;

    try {
        if (!processImage)
            throw new Error("The background .NET runtime did not start.");

        const output = processImage(requestId, sourceBytes);
        const transferable = output instanceof Uint8Array ? output : new Uint8Array(output);
        self.postMessage({ type: "result", requestId, output: transferable }, [transferable.buffer]);
    }
    catch (error) {
        self.postMessage({
            type: "error",
            requestId,
            error: error instanceof Error ? error.message : String(error)
        });
    }
});
