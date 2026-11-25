using System.Buffers.Binary;
using System.IO.Compression;
using System.Text.Json;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Forms;
using Microsoft.JSInterop;
using Inlay.Detection;
using Inlay.Processing;

namespace Inlay.Web.Pages;

public partial class Home : IAsyncDisposable
{
    private const long MaximumFileSize = 25 * 1024 * 1024;
    private const long MaximumBatchSize = 100 * 1024 * 1024;
    private const int MaximumFiles = 50;

    private const int MaximumConcurrentWorkers = 2;
    private readonly List<BatchImage> images = [];
    private IJSObjectReference? workerPoolModule;
    private DotNetObjectReference<Home>? workerProgressReference;
    private int workerCount;
    private int selectionVersion;
    private string processingMessage = "";
    private BatchImage? activeImage;
    private string error = "";
    private string validationMessage = "";
    private bool isProcessing;
    private bool isDarkTheme;
    private int sourceWidth;
    private int sourceHeight;
    private int cropX;
    private int cropY;
    private int cropWidth;
    private int cropHeight;
    private int cornerRadius;
    private ElementReference cropStage;
    private ElementReference sourceImageElement;
    private ElementReference cropSelection;
    private ElementReference cropPreviewCanvas;
    private ElementReference cropReadout;
    private ElementReference cornerRadiusSlider;
    private ElementReference cornerRadiusValue;
    private DotNetObjectReference<Home>? cropEditorReference;
    private bool cropEditorNeedsInitialization = true;

    private string StatusClass => GetStatusClass(activeImage?.Detection);

    protected override async Task OnAfterRenderAsync(bool firstRender)
    {
        if (firstRender)
        {
            isDarkTheme = await JS.InvokeAsync<bool>("inlay.isDarkMode");
            StateHasChanged();
        }

        if (activeImage is null || cropWidth <= 0 || cropHeight <= 0 || !cropEditorNeedsInitialization)
            return;

        cropEditorNeedsInitialization = false;
        cropEditorReference ??= DotNetObjectReference.Create(this);
        await JS.InvokeVoidAsync(
            "inlayCropEditor.initialize",
            cropStage,
            sourceImageElement,
            cropSelection,
            cropPreviewCanvas,
            cropReadout,
            cornerRadiusSlider,
            cornerRadiusValue,
            cropEditorReference,
            sourceWidth,
            sourceHeight,
            cropX,
            cropY,
            cropWidth,
            cropHeight,
            cornerRadius);
    }

    [JSInvokable]
    public Task CommitCropFromEditor(int x, int y, int width, int height, int radius)
    {
        cropX = x;
        cropY = y;
        cropWidth = width;
        cropHeight = height;
        cornerRadius = radius;
        ApplyCrop();
        StateHasChanged();
        return Task.CompletedTask;
    }

    public async ValueTask DisposeAsync()
    {
        cropEditorReference?.Dispose();
        workerProgressReference?.Dispose();

        if (workerPoolModule is not null)
        {
            try
            {
                await workerPoolModule.InvokeVoidAsync("dispose");
                await workerPoolModule.DisposeAsync();
            }
            catch (JSDisconnectedException)
            {
            }
        }

        foreach (var image in images)
        {
            if (string.IsNullOrEmpty(image.SourcePreview))
                continue;

            try
            {
                await JS.InvokeVoidAsync("inlay.revokeObjectUrl", image.SourcePreview);
            }
            catch (JSDisconnectedException)
            {
                break;
            }
        }
    }

    private async Task ToggleTheme()
    {
        isDarkTheme = await JS.InvokeAsync<bool>("inlay.toggleTheme");
    }
    private int ReadyCount => images.Count(image => image.Status == BatchStatus.Completed && image.Detection?.Status == DetectionStatus.Extracted);
    private int ReviewCount => images.Count(image => image.Status == BatchStatus.Completed && image.Detection?.Status == DetectionStatus.Review);
    private int RejectedCount => images.Count(image => image.Status == BatchStatus.Completed && image.Detection?.Status == DetectionStatus.Rejected);
    private int FailedCount => images.Count(image => image.Status == BatchStatus.Failed);
    private bool HasResults => images.Count > 0;

    private static string GetStatusClass(DetectionResult? result) => result?.Status switch
    {
        DetectionStatus.Extracted => "success",
        DetectionStatus.Review => "review",
        _ => "rejected"
    };

    private static string GetStatusLabel(BatchImage image)
    {
        if (image.Status == BatchStatus.Failed)
            return "Failed";

        return image.Status switch
        {
            BatchStatus.Queued => "Queued",
            BatchStatus.Reading => "Reading",
            BatchStatus.Decoding => "Decoding image",
            BatchStatus.Detecting => "Finding image",
            BatchStatus.CreatingPreview => "Preparing preview",
            BatchStatus.Completed when image.Detection?.Status == DetectionStatus.Extracted => "Extracted",
            BatchStatus.Completed when image.Detection?.Status == DetectionStatus.Review => "Review",
            BatchStatus.Completed when image.Detection is { Status: DetectionStatus.Rejected, Best: null } => "No crop",
            BatchStatus.Completed when image.Detection?.Status == DetectionStatus.Rejected => "Rejected",
            BatchStatus.Completed => "Manual crop",
            _ => "Queued"
        };
    }

    private async Task LoadImages(InputFileChangeEventArgs args)
    {
        if (isProcessing)
            return;

        error = "";
        validationMessage = "";

        IReadOnlyList<IBrowserFile> files;
        try
        {
            files = args.GetMultipleFiles(MaximumFiles);
        }
        catch (InvalidOperationException)
        {
            error = $"Choose no more than {MaximumFiles} images at a time.";
            return;
        }

        if (files.Sum(file => file.Size) > MaximumBatchSize)
        {
            error = "The selected files exceed the 100 MB total batch limit. Choose fewer or smaller images.";
            return;
        }

        foreach (var image in images)
        {
            if (!string.IsNullOrEmpty(image.SourcePreview))
                await JS.InvokeVoidAsync("inlay.revokeObjectUrl", image.SourcePreview);
        }

        images.Clear();
        activeImage = null;
        cropEditorNeedsInitialization = true;
        selectionVersion++;
        foreach (var file in files)
            images.Add(new BatchImage { FileName = file.Name });

        if (images.Count == 0)
            return;

        isProcessing = true;
        processingMessage = "Starting background image workers…";
        await RefreshBatchAsync();

        try
        {
            await EnsureWorkerPoolAsync();
            processingMessage = $"Processing up to {workerCount} images at a time in the background. Completed crops remain selectable.";
            await RefreshBatchAsync();

            using var concurrency = new SemaphoreSlim(workerCount, workerCount);
            var tasks = files.Select((file, index) =>
                ProcessWithLimitAsync(file, images[index], concurrency));
            await Task.WhenAll(tasks);
        }
        catch (Exception exception)
        {
            error = $"Background processing could not continue: {exception.Message}";
        }
        finally
        {
            isProcessing = false;
            processingMessage = "";
            await RefreshBatchAsync();
        }
    }

    private async Task EnsureWorkerPoolAsync()
    {
        workerPoolModule ??= await JS.InvokeAsync<IJSObjectReference>("import", "./worker-pool.js");
        workerProgressReference ??= DotNetObjectReference.Create(this);
        workerCount = await workerPoolModule.InvokeAsync<int>(
            "initialize", MaximumConcurrentWorkers, workerProgressReference);

        if (workerCount < 1)
            throw new InvalidOperationException("No background image workers could be started.");
    }

    private async Task ProcessWithLimitAsync(IBrowserFile file, BatchImage image, SemaphoreSlim concurrency)
    {
        await concurrency.WaitAsync();
        try
        {
            await ProcessImage(file, image);
        }
        finally
        {
            concurrency.Release();
        }
    }

    [JSInvokable]
    public Task ReportWorkerProgress(string imageId, int progress)
    {
        var image = images.FirstOrDefault(item => item.Id == imageId);
        if (image is null || image.Status != BatchStatus.Detecting)
            return Task.CompletedTask;

        image.Progress = Math.Clamp(18 + progress * 62 / 100, 18, 80);
        return Task.CompletedTask;
    }

    private async Task ProcessImage(IBrowserFile file, BatchImage image)
    {
        if (file.Size > MaximumFileSize)
        {
            FailImage(image, "Larger than 25 MB.");
            return;
        }

        if (file.ContentType is not ("image/png" or "image/jpeg" or "image/webp"))
        {
            FailImage(image, "Unsupported format.");
            return;
        }

        image.Status = BatchStatus.Reading;
        image.Progress = 0;

        try
        {
            byte[] sourceBytes;
            await using (var stream = file.OpenReadStream(MaximumFileSize))
            using (var memory = new MemoryStream())
            {
                var buffer = new byte[64 * 1024];
                var nextProgressUpdate = 5;

                while (true)
                {
                    var bytesRead = await stream.ReadAsync(buffer.AsMemory());
                    if (bytesRead == 0)
                        break;

                    memory.Write(buffer, 0, bytesRead);
                    var progress = file.Size == 0 ? 100 : (int)(memory.Length * 100L / file.Size);
                    if (progress >= nextProgressUpdate || progress == 100)
                    {
                        image.Progress = Math.Clamp(progress * 15 / 100, 0, 15);
                        nextProgressUpdate = progress + 5;
                    }
                }

                sourceBytes = memory.ToArray();
            }

            image.SourceBytes = sourceBytes;
            image.SourceContentType = file.ContentType;
            image.Status = BatchStatus.Detecting;
            image.Progress = 18;

            var payload = await workerPoolModule!.InvokeAsync<byte[]>("process", image.Id, sourceBytes);
            var processed = ParseWorkerPayload(payload);

            image.SourceWidth = processed.Metadata.SourceWidth;
            image.SourceHeight = processed.Metadata.SourceHeight;
            image.Detection = new DetectionResult(
                Enum.Parse<DetectionStatus>(processed.Metadata.Status),
                CreateCandidate(processed.Metadata.Best),
                CreateCandidate(processed.Metadata.Second),
                processed.Metadata.Reason);

            var best = image.Detection.Best;
            image.CropX = best?.Bounds.X ?? 0;
            image.CropY = best?.Bounds.Y ?? 0;
            image.CropWidth = best?.Bounds.Width ?? image.SourceWidth;
            image.CropHeight = best?.Bounds.Height ?? image.SourceHeight;
            image.CornerRadius = best?.CornerRadius ?? 0;

            image.Status = BatchStatus.CreatingPreview;
            image.Progress = 94;
            image.CropBytes = processed.CropBytes;
            image.ThumbnailPreview = CreateDataUrl(processed.ThumbnailBytes, "image/png");
            image.Status = BatchStatus.Completed;
            image.Progress = 100;

            if (activeImage is null)
                await SelectImage(image);
        }
        catch (Exception exception)
        {
            FailImage(image, $"Could not process: {exception.Message}");
            image.SourceBytes = null;
            image.CropBytes = null;
            image.SourcePreview = "";
            image.ThumbnailPreview = "";
        }
    }

    private static ProcessedImage ParseWorkerPayload(byte[] payload)
    {
        if (payload.Length < 12)
            throw new InvalidDataException("The background processor returned an incomplete result.");

        var metadataLength = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(0, 4));
        var cropLength = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(4, 4));
        var thumbnailLength = BinaryPrimitives.ReadInt32LittleEndian(payload.AsSpan(8, 4));
        var expectedLength = 12L + metadataLength + cropLength + thumbnailLength;

        if (metadataLength <= 0 || cropLength <= 0 || thumbnailLength <= 0 || expectedLength != payload.LongLength)
            throw new InvalidDataException("The background processor returned invalid image data.");

        var metadata = JsonSerializer.Deserialize<WorkerMetadata>(
            payload.AsSpan(12, metadataLength),
            new JsonSerializerOptions(JsonSerializerDefaults.Web));

        if (metadata is null || metadata.SourceWidth <= 0 || metadata.SourceHeight <= 0)
            throw new InvalidDataException("The background processor returned invalid image metadata.");

        var cropOffset = 12 + metadataLength;
        var thumbnailOffset = cropOffset + cropLength;
        return new ProcessedImage(
            metadata,
            payload.AsSpan(cropOffset, cropLength).ToArray(),
            payload.AsSpan(thumbnailOffset, thumbnailLength).ToArray());
    }

    private static DetectionCandidate? CreateCandidate(WorkerCandidate? candidate)
    {
        if (candidate is null)
            return null;

        return new DetectionCandidate(
            new OpenCvSharp.Rect(candidate.X, candidate.Y, candidate.Width, candidate.Height),
            candidate.Confidence,
            candidate.BoundaryContinuity,
            candidate.EdgeStrength,
            candidate.GeometryFit,
            candidate.SharpnessContrast,
            candidate.CornerFit,
            candidate.CornerRadius,
            candidate.AreaRatio);
    }

    private static void FailImage(BatchImage image, string message)
    {
        image.Error = message;
        image.Status = BatchStatus.Failed;
        image.Progress = 100;
    }

    private Task RefreshBatchAsync() => InvokeAsync(StateHasChanged);

    private async Task SelectImage(BatchImage image)
    {
        if (!image.CanSelect)
            return;

        var requestVersion = ++selectionVersion;
        SaveActiveImage();

        foreach (var other in images)
        {
            if (ReferenceEquals(other, image) || string.IsNullOrEmpty(other.SourcePreview))
                continue;

            await JS.InvokeVoidAsync("inlay.revokeObjectUrl", other.SourcePreview);
            other.SourcePreview = "";
        }

        activeImage = image;
        sourceWidth = image.SourceWidth;
        sourceHeight = image.SourceHeight;
        cropX = image.CropX;
        cropY = image.CropY;
        cropWidth = image.CropWidth;
        cropHeight = image.CropHeight;
        cornerRadius = image.CornerRadius;
        cropEditorNeedsInitialization = true;
        validationMessage = "";

        if (string.IsNullOrEmpty(image.SourcePreview))
        {
            var previewUrl = await JS.InvokeAsync<string>(
                "inlay.createObjectUrl", image.SourceBytes!, image.SourceContentType);

            if (requestVersion != selectionVersion || !ReferenceEquals(activeImage, image))
            {
                await JS.InvokeVoidAsync("inlay.revokeObjectUrl", previewUrl);
                return;
            }

            image.SourcePreview = previewUrl;
        }

        await InvokeAsync(StateHasChanged);
    }

    private void SaveActiveImage()
    {
        if (activeImage is null)
            return;

        activeImage.CropX = cropX;
        activeImage.CropY = cropY;
        activeImage.CropWidth = cropWidth;
        activeImage.CropHeight = cropHeight;
        activeImage.CornerRadius = cornerRadius;
    }

    private void UseDetectionBounds()
    {
        if (activeImage?.Detection?.Best is null)
            return;

        cropX = activeImage.Detection.Best.Bounds.X;
        cropY = activeImage.Detection.Best.Bounds.Y;
        cropWidth = activeImage.Detection.Best.Bounds.Width;
        cropHeight = activeImage.Detection.Best.Bounds.Height;
        cornerRadius = activeImage.Detection.Best.CornerRadius;
        cropEditorNeedsInitialization = true;
        ApplyCrop();
    }

    private void ApplyCrop()
    {
        validationMessage = "";
        if (activeImage?.SourceBytes is null)
            return;

        activeImage.CropBytes = null;
        activeImage.ThumbnailPreview = "";

        if (cropX < 0 || cropY < 0 || cropWidth <= 0 || cropHeight <= 0 ||
            cropX >= sourceWidth || cropY >= sourceHeight ||
            (long)cropX + cropWidth > sourceWidth || (long)cropY + cropHeight > sourceHeight)
        {
            validationMessage = "The crop rectangle must fit inside the original image.";
            return;
        }

        if (cornerRadius < 0 || cornerRadius > Math.Min(cropWidth, cropHeight) / 2)
        {
            validationMessage = "The corner radius is outside the allowed range.";
            return;
        }

        try
        {
            activeImage.CropBytes = ImageProcessing.CreateCrop(activeImage.SourceBytes!, cropX, cropY, cropWidth, cropHeight, cornerRadius);
            activeImage.ThumbnailPreview = CreateDataUrl(ImageProcessing.CreateThumbnail(activeImage.CropBytes), "image/png");
            SaveActiveImage();
            cropEditorNeedsInitialization = true;
        }
        catch (Exception exception)
        {
            activeImage.CropBytes = null;
            activeImage.ThumbnailPreview = "";
            validationMessage = $"Could not create crop: {exception.Message}";
        }
    }

    private static string CreateDataUrl(byte[] bytes, string contentType) =>
        $"data:{contentType};base64,{Convert.ToBase64String(bytes)}";

    private async Task DownloadBatchDiagnostics()
    {
        if (images.Count == 0)
            return;

        var report = new
        {
            format = "inlay-batch-diagnostics",
            version = 1,
            thresholds = new
            {
                automatic = DetectionResult.AutomaticThreshold,
                review = DetectionResult.ReviewThreshold
            },
            summary = new
            {
                total = images.Count,
                ready = ReadyCount,
                review = ReviewCount,
                rejected = RejectedCount,
                failed = FailedCount
            },
            images = images.Select((image, index) =>
            {
                var detection = image.Detection;
                var detectedBounds = detection?.Best?.Bounds;

                return new
                {
                    index = index + 1,
                    processingStatus = image.Status.ToString(),
                    source = image.SourceWidth > 0 && image.SourceHeight > 0
                        ? new { width = image.SourceWidth, height = image.SourceHeight }
                        : null,
                    detection = detection is null
                        ? null
                        : new
                        {
                            status = detection.Status.ToString(),
                            reason = detection.Reason,
                            best = CreateCandidateDiagnostics(detection.Best),
                            second = CreateCandidateDiagnostics(detection.Second)
                        },
                    currentCrop = image.CropWidth > 0 && image.CropHeight > 0
                        ? new
                        {
                            x = image.CropX,
                            y = image.CropY,
                            width = image.CropWidth,
                            height = image.CropHeight,
                            right = image.CropX + image.CropWidth,
                            bottom = image.CropY + image.CropHeight,
                            cornerRadius = image.CornerRadius
                        }
                        : null,
                    adjustmentFromDetected = detectedBounds is { } bounds
                        ? new
                        {
                            left = image.CropX - bounds.X,
                            top = image.CropY - bounds.Y,
                            right = image.CropX + image.CropWidth - bounds.Right,
                            bottom = image.CropY + image.CropHeight - bounds.Bottom
                        }
                        : null
                };
            }).ToArray()
        };

        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web)
        {
            WriteIndented = true
        };
        var bytes = JsonSerializer.SerializeToUtf8Bytes(report, options);
        await Download(bytes, "inlay-batch-diagnostics.json", "application/json");
    }

    private static object? CreateCandidateDiagnostics(DetectionCandidate? candidate)
    {
        if (candidate is null)
            return null;

        var bounds = candidate.Bounds;
        return new
        {
            bounds = new
            {
                x = bounds.X,
                y = bounds.Y,
                width = bounds.Width,
                height = bounds.Height,
                right = bounds.Right,
                bottom = bounds.Bottom
            },
            confidence = candidate.Confidence,
            boundaryContinuity = candidate.BoundaryContinuity,
            edgeStrength = candidate.EdgeStrength,
            geometryFit = candidate.GeometryFit,
            sharpnessContrast = candidate.SharpnessContrast,
            cornerFit = candidate.CornerFit,
            cornerRadius = candidate.CornerRadius,
            areaRatio = candidate.AreaRatio
        };
    }

    private async Task DownloadCrop()
    {
        if (activeImage?.CropBytes is null)
            return;

        await Download(activeImage.CropBytes, GetOutputName(activeImage.FileName), "image/png");
    }

    private async Task DownloadAll()
    {
        var ready = images.Where(image => image.CropBytes is not null).ToList();
        if (ready.Count == 0)
            return;

        using var archiveStream = new MemoryStream();
        using (var archive = new ZipArchive(archiveStream, ZipArchiveMode.Create, leaveOpen: true))
        {
            var names = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var image in ready)
            {
                var baseName = GetOutputName(image.FileName);
                var outputName = baseName;
                var suffix = 2;
                while (!names.Add(outputName))
                    outputName = $"{Path.GetFileNameWithoutExtension(baseName)}-{suffix++}.png";

                var entry = archive.CreateEntry(outputName, CompressionLevel.Fastest);
                await using var entryStream = entry.Open();
                await entryStream.WriteAsync(image.CropBytes!);
            }
        }

        await Download(archiveStream.ToArray(), "inlay-extractions.zip", "application/zip");
    }

    private async Task Download(byte[] bytes, string fileName, string contentType)
    {
        await using var stream = new MemoryStream(bytes);
        using var streamReference = new DotNetStreamReference(stream);
        await JS.InvokeVoidAsync("inlay.download", fileName, streamReference, contentType);
    }

    private static string GetOutputName(string fileName) =>
        $"{Path.GetFileNameWithoutExtension(fileName)}.inlay.png";

    private sealed record WorkerMetadata(
        int SourceWidth,
        int SourceHeight,
        string Status,
        string Reason,
        WorkerCandidate? Best,
        WorkerCandidate? Second);

    private sealed record WorkerCandidate(
        int X,
        int Y,
        int Width,
        int Height,
        double Confidence,
        double BoundaryContinuity,
        double EdgeStrength,
        double GeometryFit,
        double SharpnessContrast,
        double CornerFit,
        int CornerRadius,
        double AreaRatio);

    private sealed record ProcessedImage(
        WorkerMetadata Metadata,
        byte[] CropBytes,
        byte[] ThumbnailBytes);
}
