using System.IO.Compression;
using System.Text.Json;
using Microsoft.AspNetCore.Components;
using Microsoft.AspNetCore.Components.Forms;
using Microsoft.JSInterop;
using OpenCvSharp;
using Inlay.Detection;
using Inlay.Extraction;

namespace Inlay.Web.Pages;

public partial class Home : IDisposable
{
    private const long MaximumFileSize = 25 * 1024 * 1024;
    private const long MaximumBatchSize = 100 * 1024 * 1024;
    private const int MaximumFiles = 50;

    private readonly ImageDetector detector = new();
    private readonly List<BatchImage> images = [];
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

    public void Dispose()
    {
        cropEditorReference?.Dispose();
    }

    private async Task ToggleTheme()
    {
        isDarkTheme = await JS.InvokeAsync<bool>("inlay.toggleTheme");
    }
    private int SuccessfulCount => images.Count(image => image.CropBytes is not null);
    private int ReviewCount => images.Count(image => image.Status == BatchStatus.Completed && image.Detection?.Status != DetectionStatus.Extracted);
    private int FailedCount => images.Count(image => image.Status == BatchStatus.Failed);
    private int CompletedCount => images.Count(image => image.IsComplete);
    private int BatchProgressPercent => images.Count == 0 ? 0 : (int)Math.Round(CompletedCount * 100d / images.Count);
    private string BatchProgressStyle => $"width: {BatchProgressPercent}%";
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
            BatchStatus.Detecting => "Detecting",
            BatchStatus.CreatingPreview => "Preparing preview",
            BatchStatus.Completed when image.Detection?.Status == DetectionStatus.Extracted => "Extracted",
            BatchStatus.Completed when image.Detection?.Best is not null => "Review",
            BatchStatus.Completed => "Manual crop",
            _ => "Queued"
        };
    }

    private static string GetBatchStatusClass(BatchImage image)
    {
        if (image.Status == BatchStatus.Failed)
            return "rejected";

        if (image.Status != BatchStatus.Completed)
            return "processing";

        return image.Detection?.Status == DetectionStatus.Extracted ? "success" : "review";
    }

    private static string GetProgressText(BatchImage image) => image.Status switch
    {
        BatchStatus.Queued => "Waiting",
        BatchStatus.Reading => $"Reading file · {image.Progress}%",
        BatchStatus.Detecting => "Detecting image",
        BatchStatus.CreatingPreview => "Creating preview",
        BatchStatus.Completed => GetStatusLabel(image),
        BatchStatus.Failed => image.Error,
        _ => ""
    };

    private static string GetProgressStyle(BatchImage image) =>
        $"width: {(image.IsIndeterminate ? 34 : image.Progress)}%";

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

        images.Clear();
        activeImage = null;
        cropEditorNeedsInitialization = true;
        foreach (var file in files)
            images.Add(new BatchImage { FileName = file.Name });

        if (images.Count == 0)
            return;

        isProcessing = true;
        await RefreshBatchAsync();

        try
        {
            for (var index = 0; index < files.Count; index++)
                await ProcessImage(files[index], images[index]);
        }
        finally
        {
            isProcessing = false;
            await RefreshBatchAsync();
        }
    }

    private async Task ProcessImage(IBrowserFile file, BatchImage image)
    {
        if (file.Size > MaximumFileSize)
        {
            FailImage(image, "Larger than 25 MB.");
            await RefreshBatchAsync();
            return;
        }

        if (file.ContentType is not ("image/png" or "image/jpeg" or "image/webp"))
        {
            FailImage(image, "Unsupported format.");
            await RefreshBatchAsync();
            return;
        }

        image.Status = BatchStatus.Reading;
        image.Progress = 0;
        await RefreshBatchAsync();

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
                        image.Progress = Math.Clamp(progress, 0, 100);
                        nextProgressUpdate = image.Progress + 5;
                        await RefreshBatchAsync();
                    }
                }

                sourceBytes = memory.ToArray();
            }

            image.SourceBytes = sourceBytes;
            image.SourceContentType = file.ContentType;
            image.Status = BatchStatus.Detecting;
            image.IsIndeterminate = true;
            await RefreshBatchAsync();

            using (var source = Cv2.ImDecode(image.SourceBytes, ImreadModes.Color))
            {
                if (source.Empty())
                    throw new InvalidDataException("Could not decode this image.");

                image.SourceWidth = source.Width;
                image.SourceHeight = source.Height;
                image.Detection = detector.Detect(source);

                if (image.Detection.Best is not null)
                {
                    image.CropX = image.Detection.Best.Bounds.X;
                    image.CropY = image.Detection.Best.Bounds.Y;
                    image.CropWidth = image.Detection.Best.Bounds.Width;
                    image.CropHeight = image.Detection.Best.Bounds.Height;
                    image.CornerRadius = image.Detection.Best.CornerRadius;
                }
                else
                {
                    image.CropWidth = source.Width;
                    image.CropHeight = source.Height;
                }
            }

            image.Status = BatchStatus.CreatingPreview;
            await RefreshBatchAsync();

            image.CropBytes = CreateCrop(image, image.CropX, image.CropY, image.CropWidth, image.CropHeight, image.CornerRadius);
            image.ThumbnailPreview = CreateThumbnail(image.CropBytes);
            image.Status = BatchStatus.Completed;
            image.Progress = 100;
            image.IsIndeterminate = false;

            if (activeImage is null)
                SelectImage(image);

            await RefreshBatchAsync();
        }
        catch (Exception exception)
        {
            FailImage(image, $"Could not process: {exception.Message}");
            image.SourceBytes = null;
            image.CropBytes = null;
            image.SourcePreview = "";
            image.ThumbnailPreview = "";
            await RefreshBatchAsync();
        }
    }

    private static void FailImage(BatchImage image, string message)
    {
        image.Error = message;
        image.Status = BatchStatus.Failed;
        image.Progress = 100;
        image.IsIndeterminate = false;
    }

    private async Task RefreshBatchAsync()
    {
        await InvokeAsync(StateHasChanged);
        await Task.Yield();
    }

    private void SelectImage(BatchImage image)
    {
        foreach (var other in images)
        {
            if (ReferenceEquals(other, image))
                continue;

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
        image.SourcePreview = CreateDataUrl(image.SourceBytes!, image.SourceContentType);
        cropEditorNeedsInitialization = true;
        validationMessage = "";
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
            activeImage.CropBytes = CreateCrop(activeImage, cropX, cropY, cropWidth, cropHeight, cornerRadius);
            activeImage.ThumbnailPreview = CreateThumbnail(activeImage.CropBytes);
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

    private static byte[] CreateCrop(BatchImage image, int x, int y, int width, int height, int radius)
    {
        using var source = Cv2.ImDecode(image.SourceBytes!, ImreadModes.Color);
        using var crop = new Mat(source, new Rect(x, y, width, height));
        using var bgra = new Mat();
        Cv2.CvtColor(crop, bgra, ColorConversionCodes.BGR2BGRA);
        using var mask = ImageExtractor.CreateRoundedMask(width, height, radius);
        Cv2.InsertChannel(mask, bgra, 3);

        if (!Cv2.ImEncode(".png", bgra, out var encodedCrop))
            throw new InvalidOperationException("Could not encode the crop as PNG.");

        return encodedCrop;
    }

    private static string CreateThumbnail(byte[] cropBytes)
    {
        using var source = Cv2.ImDecode(cropBytes, ImreadModes.Unchanged);
        if (source.Empty())
            throw new InvalidOperationException("Could not decode the crop thumbnail.");

        const int maximumDimension = 144;
        var scale = Math.Min(1d, maximumDimension / (double)Math.Max(source.Width, source.Height));
        using var thumbnail = new Mat();

        if (scale < 1)
        {
            var size = new Size(
                Math.Max(1, (int)Math.Round(source.Width * scale)),
                Math.Max(1, (int)Math.Round(source.Height * scale)));
            Cv2.Resize(source, thumbnail, size, interpolation: InterpolationFlags.Area);
        }
        else
        {
            source.CopyTo(thumbnail);
        }

        if (!Cv2.ImEncode(".png", thumbnail, out var encodedThumbnail))
            throw new InvalidOperationException("Could not encode the crop thumbnail.");

        return CreateDataUrl(encodedThumbnail, "image/png");
    }

    private static string CreateDataUrl(byte[] bytes, string contentType) =>
        $"data:{contentType};base64,{Convert.ToBase64String(bytes)}";

    private async Task DownloadDiagnostics()
    {
        if (activeImage is null)
            return;

        var detection = activeImage.Detection;
        var detectedBounds = detection?.Best?.Bounds;
        var currentRight = cropX + cropWidth;
        var currentBottom = cropY + cropHeight;

        var report = new
        {
            format = "inlay-diagnostics",
            version = 1,
            source = new
            {
                width = sourceWidth,
                height = sourceHeight
            },
            detection = new
            {
                status = detection?.Status.ToString(),
                reason = detection?.Reason,
                automaticThreshold = DetectionResult.AutomaticThreshold,
                reviewThreshold = DetectionResult.ReviewThreshold,
                best = CreateCandidateDiagnostics(detection?.Best),
                second = CreateCandidateDiagnostics(detection?.Second)
            },
            currentCrop = new
            {
                x = cropX,
                y = cropY,
                width = cropWidth,
                height = cropHeight,
                right = currentRight,
                bottom = currentBottom,
                cornerRadius
            },
            adjustmentFromDetected = detectedBounds is { } bounds
                ? new
                {
                    left = cropX - bounds.X,
                    top = cropY - bounds.Y,
                    right = currentRight - bounds.Right,
                    bottom = currentBottom - bounds.Bottom
                }
                : null
        };

        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web)
        {
            WriteIndented = true
        };
        var bytes = JsonSerializer.SerializeToUtf8Bytes(report, options);
        await Download(bytes, "inlay-diagnostics.json", "application/json");
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

    private enum BatchStatus
    {
        Queued,
        Reading,
        Detecting,
        CreatingPreview,
        Completed,
        Failed
    }

    private sealed class BatchImage
    {
        public string FileName { get; init; } = "";
        public string Error { get; set; } = "";
        public byte[]? SourceBytes { get; set; }
        public byte[]? CropBytes { get; set; }
        public string SourceContentType { get; set; } = "image/png";
        public string SourcePreview { get; set; } = "";
        public string ThumbnailPreview { get; set; } = "";
        public int SourceWidth { get; set; }
        public int SourceHeight { get; set; }
        public int CropX { get; set; }
        public int CropY { get; set; }
        public int CropWidth { get; set; }
        public int CropHeight { get; set; }
        public int CornerRadius { get; set; }
        public int Progress { get; set; }
        public bool IsIndeterminate { get; set; }
        public BatchStatus Status { get; set; } = BatchStatus.Queued;
        public bool IsComplete => Status is BatchStatus.Completed or BatchStatus.Failed;
        public bool CanSelect => Status == BatchStatus.Completed && SourceBytes is not null && SourceWidth > 0 && SourceHeight > 0;
        public DetectionResult? Detection { get; set; }
    }
}
